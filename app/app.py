import gradio as gr
import os
import sys
import json
import tempfile
import shutil
import subprocess
import logging
import traceback
import platform
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import io

# ============================================================
# LOGGING SETUP — errors written to logs/app.log + stderr
# ============================================================

_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_dir / "app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("lover-clinic")
logger.info("=== Lover Clinic AI Video Tools starting ===")

# ============================================================
# COMPATIBILITY PATCH — torchvision >= 0.16 removed functional_tensor
# (needed by basicsr / realesrgan)
# Also catches OSError for Windows CUDA DLL load failures
# ============================================================

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except (ModuleNotFoundError, OSError):
    try:
        import torchvision.transforms.functional as _ft
        sys.modules["torchvision.transforms.functional_tensor"] = _ft
        logger.info("Applied torchvision.functional_tensor compatibility patch")
    except (ImportError, OSError) as _e:
        logger.warning(f"Could not apply torchvision patch: {_e}")

# ============================================================
# BASE64 IMAGE HELPERS — embed images directly to avoid /file= issues
# ============================================================

def _b64_img(filename: str) -> str:
    """Return a base64 data-URI for an image in the static/ folder."""
    try:
        import base64
        p = Path(__file__).parent / "static" / filename
        data = base64.b64encode(p.read_bytes()).decode()
        ext = p.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
        return f"data:{mime};base64,{data}"
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
        return ""

_ICON_SRC = _b64_img("icon.png")
_LOGO_SRC = _b64_img("logo.png")
logger.info(f"Branding loaded — icon: {len(_ICON_SRC)>0}, logo: {len(_LOGO_SRC)>0}")

# ============================================================
# PROJECT PATHS
# ============================================================
_PROJ_ROOT  = Path(__file__).parent.parent          # project root (above app/)
_OUTPUT_ROOT = _PROJ_ROOT / "output"
_OUTPUT_ROOT.mkdir(exist_ok=True)
DEFAULT_OUT = str(_OUTPUT_ROOT)

# ============================================================
# SETTINGS PERSISTENCE
# ============================================================

_SETTINGS_FILE = _PROJ_ROOT / "lc_settings.json"

_SETTINGS_DEFAULTS: dict = {
    # Photo Upscale
    "up_scale":   4,
    "up_model":   "General Photo",
    "up_fmt":     "PNG",
    "up_out_dir": None,   # filled lazily (output dir may not exist yet)
    # Video Upscale
    "vid_scale":   2,
    "vid_model":   "General",
    "vid_out_dir": None,
    # Remove BG
    "bg_model":   "BiRefNet — General (Best)",
    "bg_option":  "Transparent",
    "bg_fmt":     "PNG",
    "bg_out_dir": None,
    # Enhance
    "enh_scale":   2,
    "enh_bg":      True,
    "enh_fmt":     "PNG",
    "enh_out_dir": None,
    # Resize
    "res_w":       1920,
    "res_h":       1080,
    "res_ar":      True,
    "res_filter":  "Lanczos (Best Quality)",
    "res_fmt":     "PNG",
    "res_out_dir": None,
    # Crop
    "crop_fmt":     "PNG",
    "crop_out_dir": None,
    # Convert
    "conv_fmt":     "WEBP",
    "conv_q":       95,
    "conv_out_dir": None,
}


def _load_settings() -> dict:
    """Load persisted settings, falling back to defaults for missing keys."""
    cfg = _SETTINGS_DEFAULTS.copy()
    if _SETTINGS_FILE.exists():
        try:
            saved = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in saved.items() if k in cfg})
        except Exception as e:
            logger.warning(f"Could not read settings: {e}")
    # fill None dirs with real defaults
    _dir_defaults = {
        "up_out_dir":   "photo",
        "vid_out_dir":  "video",
        "bg_out_dir":   "remove_bg",
        "enh_out_dir":  "enhance",
        "res_out_dir":  "resize",
        "crop_out_dir": "crop",
        "conv_out_dir": "convert",
    }
    for key, subdir in _dir_defaults.items():
        if cfg[key] is None:
            cfg[key] = str(_OUTPUT_ROOT / subdir)
    return cfg


def _save_settings(cfg: dict):
    try:
        _SETTINGS_FILE.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not save settings: {e}")


def _make_saver(key: str):
    """Return a 1-arg Gradio handler that persists one setting key."""
    def _fn(val):
        cfg = _load_settings()
        cfg[key] = val
        _save_settings(cfg)
    return _fn

# ============================================================
# LAZY MODEL MANAGERS
# ============================================================

_realesrgan_cache = {}
_gfpgan_cache = {}
_rembg_sessions = {}


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except (ImportError, OSError) as e:
        logger.warning(f"torch unavailable ({e}), falling back to CPU")
    return "cpu"


def get_realesrgan(scale=4, model_type="general"):
    key = f"{scale}_{model_type}"
    if key not in _realesrgan_cache:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        device = get_device()

        if model_type == "anime":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.5.0/realesr-animevideov3.pth")
            tile = 400
        elif scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.1/RealESRGAN_x2plus.pth")
            tile = 512
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.1.0/RealESRGAN_x4plus.pth")
            tile = 512

        _realesrgan_cache[key] = RealESRGANer(
            scale=4 if model_type == "anime" else scale,
            model_path=model_url,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda"),
            device=device,
        )
    return _realesrgan_cache[key]


def get_rembg_session(model_name="birefnet-general"):
    if model_name not in _rembg_sessions:
        from rembg import new_session
        _rembg_sessions[model_name] = new_session(model_name)
    return _rembg_sessions[model_name]


# ============================================================
# AI FUNCTIONS
# ============================================================

def upscale_photo(image, scale, model_type, output_dir, fmt="PNG", progress=gr.Progress()):
    if image is None:
        return None, "⚠️ Please upload an image first."
    try:
        progress(0.1, desc="Loading model…")
        mt = "anime" if "Anime" in model_type else "general"
        upsampler = get_realesrgan(scale=int(scale), model_type=mt)

        progress(0.3, desc="Upscaling…")
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(img_cv2, outscale=int(scale))

        progress(0.95, desc="Finalising…")
        result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        saved = _save_image(result, output_dir, "upscaled_photo", fmt=fmt)
        return result, f"✅ Upscaled {scale}x — {result.width}×{result.height} px\n💾 {saved}"
    except Exception as e:
        logger.error(f"upscale_photo failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def upscale_video(video_path, scale, model_type, output_dir, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️ Please upload a video first."
    try:
        progress(0.05, desc="Loading model…")
        mt = "anime" if "Anime" in model_type else "general"
        upsampler = get_realesrgan(scale=int(scale), model_type=mt)

        tmpdir = tempfile.mkdtemp()
        frames_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(frames_dir)
        os.makedirs(out_dir)

        progress(0.1, desc="Reading video…")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fp = os.path.join(frames_dir, f"frame_{idx:08d}.png")
            cv2.imwrite(fp, frame)
            frames.append(fp)
            idx += 1
        cap.release()

        if not frames:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, "❌ Could not read video frames."

        out_paths = []
        for i, fp in enumerate(frames):
            progress(0.1 + 0.75 * (i / len(frames)),
                     desc=f"Processing frame {i+1}/{len(frames)}…")
            frame_bgr = cv2.imread(fp)
            out_frame, _ = upsampler.enhance(frame_bgr, outscale=int(scale))
            op = os.path.join(out_dir, f"frame_{i:08d}.png")
            cv2.imwrite(op, out_frame)
            out_paths.append(op)

        progress(0.9, desc="Assembling video…")
        raw_video = os.path.join(tmpdir, "raw.mp4")
        out_video = os.path.join(tmpdir, "output.mp4")

        # Step 1: assemble frames → raw_video with cv2 (mp4v)
        sample = cv2.imread(out_paths[0])
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter(
            raw_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )
        for op in out_paths:
            writer.write(cv2.imread(op))
        writer.release()

        # Step 2: re-encode to H.264 with ffmpeg so browser can play it
        def _find_ffmpeg():
            """Return (exe_path, env) for a working ffmpeg with libx264."""
            candidates = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Users\oomzp\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
            ]
            import shutil as _sh
            found = _sh.which("ffmpeg")
            if found:
                candidates.append(found)
            for exe in candidates:
                if not os.path.isfile(exe):
                    continue
                exe_dir = str(Path(exe).parent)
                env = os.environ.copy()
                env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
                try:
                    r = subprocess.run([exe, "-version"], capture_output=True, env=env, timeout=5)
                    if r.returncode == 0:
                        return exe, env
                except Exception:
                    pass
            return None, None

        try:
            ffmpeg_exe, ffmpeg_env = _find_ffmpeg()
            if not ffmpeg_exe:
                raise RuntimeError("No working ffmpeg found")
            subprocess.run([
                ffmpeg_exe, "-y",
                "-i", raw_video,
                "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p",
                out_video
            ], check=True, capture_output=True, env=ffmpeg_env)
            logger.info(f"ffmpeg H.264 encode OK (exe={ffmpeg_exe})")
        except Exception as fe:
            stderr = getattr(fe, "stderr", b"")
            logger.warning(f"ffmpeg H.264 encode failed: {stderr[-300:] if stderr else fe} — using raw mp4v")
            out_video = raw_video

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        final_path = str(Path(output_dir) / f"upscaled_video_{ts}.mp4")
        shutil.copy2(out_video, final_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

        progress(1.0, desc="Done!")
        logger.info(f"upscale_video done — returning path: {final_path!r}")
        return gr.update(value=final_path, visible=True), f"✅ Video upscaled {scale}x — saved to {final_path}"
    except Exception as e:
        logger.error(f"upscale_video failed: {e}\n{traceback.format_exc()}")
        return gr.update(value=None, visible=True), f"❌ Error: {e}"


def remove_background(image, model_choice, bg_option, custom_bg, output_dir, fmt="PNG",
                      progress=gr.Progress()):
    if image is None:
        return None, "⚠️ Please upload an image first."
    try:
        model_map = {
            "BiRefNet — General (Best)": "birefnet-general",
            "BiRefNet — Portrait": "birefnet-portrait",
            "U2Net": "u2net",
            "RMBG 1.4": "briarmbg",
        }
        rembg_model = model_map.get(model_choice, "birefnet-general")

        progress(0.15, desc="Loading model…")
        session = get_rembg_session(rembg_model)

        progress(0.4, desc="Removing background…")
        from rembg import remove
        result = remove(image, session=session)  # RGBA PIL image

        if bg_option == "Transparent":
            final = result
        elif bg_option == "Custom Image" and custom_bg is not None:
            bg = custom_bg.resize(result.size).convert("RGBA")
            final = Image.alpha_composite(bg, result).convert("RGB")
        else:
            color_map = {
                "White": (255, 255, 255),
                "Black": (0, 0, 0),
                "Red": (180, 0, 0),
            }
            color = color_map.get(bg_option, (255, 255, 255))
            bg = Image.new("RGBA", result.size, (*color, 255))
            final = Image.alpha_composite(bg, result).convert("RGB")

        progress(1.0)
        save_fmt = "PNG" if bg_option == "Transparent" else fmt
        saved = _save_image(final, output_dir, "removed_bg", fmt=save_fmt)
        return final, f"✅ Background removed\n💾 {saved}"
    except Exception as e:
        logger.error(f"remove_background failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def enhance_image(image, upscale_factor, enhance_bg, output_dir, fmt="PNG", progress=gr.Progress()):
    if image is None:
        return None, "⚠️ Please upload an image first."
    try:
        progress(0.1, desc="Loading GFPGAN model…")
        from gfpgan import GFPGANer

        bg_upsampler = None
        if enhance_bg:
            progress(0.15, desc="Loading background upsampler…")
            bg_upsampler = get_realesrgan(scale=2, model_type="general")

        restorer = GFPGANer(
            model_path=("https://github.com/TencentARC/GFPGAN/releases/"
                        "download/v1.3.0/GFPGANv1.3.pth"),
            upscale=int(upscale_factor),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )

        progress(0.4, desc="Enhancing image…")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _, _, output = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        progress(1.0)
        saved = _save_image(result, output_dir, "enhanced", fmt=fmt)
        return result, f"✅ Image enhanced\n💾 {saved}"
    except Exception as e:
        logger.error(f"enhance_image failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


# ============================================================
# OUTPUT FOLDER HELPERS
# ============================================================

def _output_subdir(name: str) -> str:
    """Return abs path to output/{name}/, created if needed."""
    p = _OUTPUT_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _open_folder(path: str) -> str:
    path = (path or DEFAULT_OUT).strip()
    os.makedirs(path, exist_ok=True)
    try:
        if sys.platform == "win32":
            p = path.replace("'", "''")
            subprocess.Popen([
                "powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command",
                f"Start-Process explorer.exe -ArgumentList '{p}'; "
                f"Start-Sleep -Milliseconds 500; "
                f"Add-Type -AssemblyName Microsoft.VisualBasic; "
                f"[Microsoft.VisualBasic.Interaction]::AppActivate('File Explorer')"
            ])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return f"📂 เปิด: {path}"
    except Exception as e:
        return f"❌ {e}"


def _make_cropper_html(img) -> str:
    """Return pure HTML/CSS canvas widget — NO scripts (scripts don't run via innerHTML).
    JavaScript is injected separately via .then(fn=None, js=CROP_INIT_JS)."""
    if img is None:
        return '<div style="color:#666;text-align:center;padding:32px;">⬆️ อัปโหลดรูปภาพก่อน</div>'
    import base64, io as _io
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    w, h = img.size
    return f"""
<style>
  #lc-editor{{
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    color:#e2e8f0; user-select:none;
  }}
  /* ── Toolbar ─────────────────────────────── */
  #lc-toolbar{{
    background:#0a0a0a;
    border:1px solid #222;
    border-radius:14px;
    padding:10px 14px;
    margin-bottom:8px;
    display:flex;flex-direction:column;gap:8px;
  }}
  .lc-row{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;}}
  .lc-lbl{{
    font-size:9px;font-weight:700;letter-spacing:.12em;
    text-transform:uppercase;color:#374151;
    min-width:44px;flex-shrink:0;
  }}
  /* pill button-group */
  .lc-grp{{
    display:flex;
    background:#050505;
    border:1px solid #222;
    border-radius:8px;
    overflow:hidden;
  }}
  .lc-ratio-btn{{
    background:transparent;color:#4b5563;
    border:none;border-right:1px solid #1a1a1a;
    padding:5px 11px;font-size:11px;font-weight:600;
    cursor:pointer;transition:background .18s,color .18s;
    white-space:nowrap;
  }}
  .lc-ratio-btn:last-child{{border-right:none;}}
  .lc-ratio-btn:hover{{background:#111;color:#e5e7eb;}}
  .lc-ratio-btn.lc-active{{background:#dc2626;color:#fff;}}
  /* action buttons */
  .lc-act{{
    background:#111;color:#4b5563;
    border:1px solid #222;border-radius:7px;
    padding:5px 11px;font-size:11px;font-weight:600;
    cursor:pointer;transition:all .18s;white-space:nowrap;
  }}
  .lc-act:hover{{background:#1a1a1a;color:#e5e7eb;border-color:#333;}}
  .lc-act.lc-active{{background:#16a34a;color:#fff;border-color:#15803d;}}
  /* social chips */
  .lc-chip{{
    display:inline-flex;align-items:center;gap:4px;
    background:#0a0a0a;color:#4b5563;
    border:1px solid #222;border-radius:9999px;
    padding:4px 11px;font-size:10.5px;font-weight:600;
    cursor:pointer;transition:all .18s;white-space:nowrap;
  }}
  .lc-chip:hover{{background:#111;color:#e5e7eb;border-color:#333;}}
  .lc-chip.lc-active{{background:#dc2626;color:#fff;border-color:#dc2626;}}
  .lc-sep{{width:1px;height:20px;background:#222;margin:0 3px;flex-shrink:0;}}
  /* ── Canvas area ─────────────────────────── */
  #lc-crop-wrap{{
    background:#050505;
    border:1px solid #222;
    border-radius:14px;
    padding:10px;
    display:flex;justify-content:center;align-items:center;
    overflow:auto;
  }}
  #lc-canvas{{display:block;cursor:crosshair;border-radius:4px;max-width:100%;}}
  /* ── Zoom value label ────────────────────── */
  #lc-zoom-val{{
    font-size:11px;font-weight:700;color:#4b5563;
    min-width:40px;text-align:center;letter-spacing:.02em;
  }}
  /* ── Info bar ────────────────────────────── */
  #lc-infobar{{
    display:flex;align-items:center;justify-content:space-between;
    background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;
    padding:6px 14px;margin-top:6px;
  }}
  #lc-crop-info{{font-size:11px;font-variant-numeric:tabular-nums;color:#6b7280;}}
  #lc-hint{{font-size:10px;color:#374151;}}
</style>

<div id="lc-editor">
  <div id="lc-toolbar">

    <!-- Row 1: Aspect ratio -->
    <div class="lc-row">
      <span class="lc-lbl">Ratio</span>
      <div class="lc-grp">
        <button class="lc-ratio-btn lc-active" data-ratio="free">Free</button>
        <button class="lc-ratio-btn" data-ratio="1:1">1:1</button>
        <button class="lc-ratio-btn" data-ratio="4:3">4:3</button>
        <button class="lc-ratio-btn" data-ratio="3:4">3:4</button>
        <button class="lc-ratio-btn" data-ratio="16:9">16:9</button>
        <button class="lc-ratio-btn" data-ratio="9:16">9:16</button>
        <button class="lc-ratio-btn" data-ratio="4:5">4:5</button>
        <button class="lc-ratio-btn" data-ratio="5:4">5:4</button>
        <button class="lc-ratio-btn" data-ratio="3:2">3:2</button>
        <button class="lc-ratio-btn" data-ratio="2:3">2:3</button>
      </div>
      <div class="lc-sep"></div>
      <button class="lc-act" id="lc-btn-swap" title="สลับ Portrait ↔ Landscape">⇄ Flip</button>
      <button class="lc-act" id="lc-btn-grid" title="Rule of Thirds">⊞ Grid</button>
      <button class="lc-act" id="lc-btn-center" title="จัดกึ่งกลาง">⊙ Center</button>
      <button class="lc-act" id="lc-btn-reset" title="ล้างการเลือก">✕ Clear</button>
      <div class="lc-sep"></div>
      <button class="lc-act" id="lc-btn-zoom-out" title="ซูมออก" style="padding:5px 9px;font-size:15px;line-height:1;">−</button>
      <span id="lc-zoom-val">100%</span>
      <button class="lc-act" id="lc-btn-zoom-in" title="ซูมเข้า" style="padding:5px 9px;font-size:15px;line-height:1;">+</button>
      <button class="lc-act" id="lc-btn-zoom-fit" title="Fit to screen">⊡ Fit</button>
    </div>

    <!-- Row 2: Social presets -->
    <div class="lc-row">
      <span class="lc-lbl">Social</span>
      <button class="lc-chip" data-ratio="4:5">📷 IG Feed</button>
      <button class="lc-chip" data-ratio="1:1">◻ Square</button>
      <button class="lc-chip" data-ratio="9:16">📱 Story / Reel</button>
      <button class="lc-chip" data-ratio="16:9">▶ YouTube</button>
      <button class="lc-chip" data-ratio="205:78">📘 FB Cover</button>
      <button class="lc-chip" data-ratio="4:1">💼 LinkedIn</button>
      <button class="lc-chip" data-ratio="2:3">📌 Pinterest</button>
      <button class="lc-chip" data-ratio="1200:628">🔗 OG Image</button>
    </div>

  </div>

  <!-- Canvas -->
  <div id="lc-crop-wrap">
    <canvas id="lc-canvas"></canvas>
    <img id="lc-crop-img" src="data:image/png;base64,{b64}" data-w="{w}" data-h="{h}" style="display:none;">
  </div>

  <!-- Info bar -->
  <div id="lc-infobar">
    <div id="lc-crop-info">⏳ กำลังเตรียม...</div>
    <div id="lc-hint">ลากเพื่อเลือก &nbsp;·&nbsp; ลากขอบปรับขนาด &nbsp;·&nbsp; ลากกลางเพื่อย้าย</div>
  </div>
</div>
"""


def _save_dir_row(default_subdir: str, label: str = "📁 Output Folder"):
    """Render a save-dir row. Returns (save_dir_textbox, save_status_textbox)."""
    with gr.Row():
        save_dir = gr.Textbox(
            value=_output_subdir(default_subdir),
            label=label, scale=5, lines=1,
        )
        open_btn = gr.Button("📂 Open Folder", variant="secondary", scale=1, min_width=130)
    save_status = gr.Textbox(label="", interactive=False, lines=1, visible=True,
                             elem_classes=["lc-status"])
    open_btn.click(fn=_open_folder, inputs=[save_dir], outputs=[save_status])
    return save_dir, save_status


def _save_image(img: Image.Image, save_dir: str, prefix: str, fmt: str = "PNG") -> str:
    """Save PIL image to save_dir with timestamp. Returns saved path."""
    os.makedirs(save_dir, exist_ok=True)
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt, ".png")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"{prefix}_{ts}{ext}")
    if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(path, fmt)
    return path


# ============================================================
# TOOLS FUNCTIONS
# ============================================================

def resize_image(image, width, height, maintain_ar, resample_filter, output_dir, fmt="PNG"):
    if image is None:
        return None, "⚠️ Please upload an image first."
    try:
        filter_map = {
            "Lanczos (Best Quality)": Image.LANCZOS,
            "Bicubic": Image.BICUBIC,
            "Bilinear": Image.BILINEAR,
            "Nearest (Fastest)": Image.NEAREST,
        }
        filt = filter_map.get(resample_filter, Image.LANCZOS)
        w, h = int(width), int(height)

        if maintain_ar:
            img_copy = image.copy()
            img_copy.thumbnail((w, h), filt)
            result = img_copy
        else:
            result = image.resize((w, h), filt)

        saved = _save_image(result, output_dir, "resized", fmt=fmt)
        return result, f"✅ Resized to {result.width}×{result.height} px\n💾 {saved}"
    except Exception as e:
        logger.error(f"resize_image failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def crop_image(image, coords_str, output_dir, fmt="PNG"):
    if image is None:
        return None, "⚠️ Please upload an image first."
    try:
        iw, ih = image.size
        if coords_str and coords_str.count(",") == 3:
            parts = [int(float(v)) for v in coords_str.split(",")]
            l, t, r, b = parts
        else:
            l, t, r, b = 0, 0, iw, ih
        l = max(0, min(l, iw - 1))
        t = max(0, min(t, ih - 1))
        r = max(l + 1, min(r, iw))
        b = max(t + 1, min(b, ih))
        result = image.crop((l, t, r, b))
        saved = _save_image(result, output_dir, "cropped", fmt=fmt)
        return result, f"✅ Cropped {r-l}×{b-t} px (from {l},{t} to {r},{b})\n💾 {saved}"
    except Exception as e:
        logger.error(f"crop_image failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def convert_format(image, out_format, quality, output_dir):
    if image is None:
        return None, None, "⚠️ Please upload an image first."
    try:
        ext_map = {
            "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
            "BMP": ".bmp", "TIFF": ".tiff",
        }
        ext = ext_map.get(out_format, ".jpg")
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"converted_{ts}{ext}")

        img = image.copy()
        if out_format == "JPEG":
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(out_path, "JPEG", quality=int(quality))
        elif out_format == "WEBP":
            img.save(out_path, "WEBP", quality=int(quality))
        elif out_format == "PNG":
            img.save(out_path, "PNG")
        elif out_format == "BMP":
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            img.save(out_path, "BMP")
        elif out_format == "TIFF":
            img.save(out_path, "TIFF")

        preview = Image.open(out_path)
        return preview, out_path, f"✅ Converted to {out_format}\n💾 {out_path}"
    except Exception as e:
        logger.error(f"convert_format failed: {e}\n{traceback.format_exc()}")
        return None, None, f"❌ Error: {e}"


# ============================================================
# THEME & CSS  (ported from lover-clinic-ai-voice2)
# ============================================================

CSS = """
/* ══════════════════════════════════════════════════════
   LOVER CLINIC — AI VIDEO TOOLS
   Design System: mirrors lover-clinic-app.vercel.app
══════════════════════════════════════════════════════ */

:root {
  --accent:      #dc2626;
  --accent-h:    #b91c1c;
  --accent-dim:  rgba(220,38,38,0.10);
  --accent-ring: rgba(220,38,38,0.22);
  --bg-base:     #050505;
  --bg-card:     #0a0a0a;
  --bg-raised:   #111111;
  --bg-hover:    #181818;
  --bd:          #222222;
  --bd-faint:    #161616;
  --tx-head:     #ffffff;
  --tx-body:     #e5e7eb;
  --tx-muted:    #6b7280;
  --tx-faint:    #374151;
  --ease:        cubic-bezier(.34,1.56,.64,1);
  --ease-std:    cubic-bezier(.4,0,.2,1);
  --dur:         0.22s;
  --r-xs:        6px;
  --r-sm:        8px;
  --r-md:        14px;
  --r-lg:        20px;
  --r-pill:      9999px;
}

*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container > .main {
    background: var(--bg-base) !important;
    color: var(--tx-body) !important;
    font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}
footer { display: none !important; }

/* ── Cards / Blocks ── */
.block, .form {
    background: var(--bg-card) !important;
    border: 1px solid var(--bd) !important;
    border-radius: var(--r-lg) !important;
    box-shadow: none !important;
}

/* ── Tab nav ── */
div[role="tablist"] {
    background: var(--bg-base) !important;
    border-bottom: 1px solid var(--bd) !important;
    padding: 0 20px !important;
    gap: 2px !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
}
div[role="tablist"]::-webkit-scrollbar { display: none; }

button[role="tab"] {
    background: transparent !important;
    color: var(--tx-muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 12px 16px 10px !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: color var(--dur) var(--ease-std),
                border-color var(--dur) var(--ease-std) !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
button[role="tab"]:hover {
    color: var(--tx-body) !important;
}
button[role="tab"][aria-selected="true"],
button[role="tab"].selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Primary Button ── */
button.primary, button[variant="primary"] {
    background: var(--accent) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    border-radius: var(--r-sm) !important;
    padding: 10px 28px !important;
    box-shadow: none !important;
    transition: background var(--dur), transform var(--dur) var(--ease) !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: var(--accent-h) !important;
    transform: translateY(-1px) !important;
}
button.primary:active, button[variant="primary"]:active {
    transform: translateY(0) !important;
    background: #991b1b !important;
}

/* ── Secondary Button ── */
button.secondary, button[variant="secondary"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--bd) !important;
    color: var(--tx-muted) !important;
    border-radius: var(--r-sm) !important;
    transition: all var(--dur) !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── Inputs / Textarea ── */
input:not([type="range"]):not([type="checkbox"]):not([type="radio"])
    :not([type="button"]):not([type="submit"]):not([type="file"]),
textarea, .scroll-hide {
    background: var(--bg-raised) !important;
    border: 1px solid var(--bd) !important;
    color: var(--tx-body) !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.88rem !important;
    transition: border-color var(--dur), box-shadow var(--dur) !important;
}
input:focus:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-ring) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: var(--tx-faint) !important; }

/* ── Sliders / Checkboxes ── */
input[type="range"]    { accent-color: var(--accent) !important; height: 4px !important; cursor: pointer !important; }
input[type="checkbox"],
input[type="radio"]    { accent-color: var(--accent) !important; width: 15px !important; height: 15px !important; }

/* ── Labels ── */
.label-wrap > span, label > span, .block > label > span {
    color: var(--tx-muted) !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Dropdown ── */
.wrap-inner, [data-testid="dropdown-component"] .wrap {
    background: var(--bg-raised) !important;
    border: 1px solid var(--bd) !important;
    border-radius: var(--r-sm) !important;
}
ul.options {
    background: var(--bg-card) !important;
    border: 1px solid var(--bd) !important;
    border-radius: var(--r-sm) !important;
}
ul.options li { color: var(--tx-body) !important; }
ul.options li:hover,
ul.options li.selected {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

/* ── Markdown / status ── */
.prose, .md { color: var(--tx-muted) !important; }
.prose p, .md p { color: var(--tx-muted) !important; line-height: 1.6 !important; }

.lc-status {
    min-height: 0 !important;
    padding: 4px 2px !important;
    font-size: 0.77rem !important;
    color: var(--tx-faint) !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.lc-status p { margin: 0 !important; }

/* ── Section heading ── */
.sec-head {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.22em !important;
    color: var(--accent) !important;
    text-transform: uppercase !important;
    padding-bottom: 12px !important;
    margin-bottom: 16px !important;
    border-bottom: 1px solid rgba(220,38,38,0.15) !important;
}

/* ── Info box ── */
.info-box {
    background: rgba(220,38,38,0.06);
    border-left: 2px solid var(--accent);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: rgba(255,160,100,0.85);
    margin-bottom: 12px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bd); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #333; }

/* ── Selection ── */
::selection { background: rgba(220,38,38,0.28) !important; color: #fff !important; }

/* ── Hide broken-image placeholders ── */
img[src=""], img[src="data:"] { display: none !important; }

/* ════════════════════════════════════════════
   RESPONSIVE — Tablet (≤ 1024px)
════════════════════════════════════════════ */
@media (max-width: 1024px) {
  .gradio-container { padding: 0 !important; }

  /* Stack rows → columns on tablet */
  .gradio-container .row {
    flex-wrap: wrap !important;
  }
  .gradio-container .row > .column {
    min-width: min(100%, 340px) !important;
    flex: 1 1 340px !important;
  }

  /* Slightly smaller tab text */
  button[role="tab"] {
    font-size: 0.68rem !important;
    padding: 10px 12px 9px !important;
  }

  /* Crop tool toolbar wraps */
  #lc-toolbar { gap: 7px !important; }
  .lc-row     { flex-wrap: wrap !important; }
}

/* ════════════════════════════════════════════
   RESPONSIVE — Mobile (≤ 640px)
════════════════════════════════════════════ */
@media (max-width: 640px) {
  /* Full-width columns */
  .gradio-container .row > .column {
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }

  /* Tabs: smaller, always scroll */
  div[role="tablist"] {
    padding: 0 8px !important;
  }
  button[role="tab"] {
    font-size: 0.62rem !important;
    padding: 9px 10px 8px !important;
    letter-spacing: 0.04em !important;
  }

  /* Primary button full-width on mobile */
  button.primary, button[variant="primary"] {
    width: 100% !important;
    padding: 12px 16px !important;
  }

  /* Blocks: tighter radius on mobile */
  .block, .form { border-radius: var(--r-md) !important; }

  /* Crop toolbar: smaller chips */
  .lc-ratio-btn { padding: 5px 8px !important; font-size: 10px !important; }
  .lc-chip      { padding: 4px 8px !important; font-size: 9.5px !important; }
  .lc-act       { padding: 5px 8px !important; font-size: 10px !important; }
  #lc-zoom-val  { min-width: 30px !important; font-size: 10px !important; }
  #lc-hint      { display: none !important; }
}
"""


def _header_html():
    import base64 as _b64

    # ── Logo ──────────────────────────────────────────────
    logo_tag = ""
    for fname in ("logo_white.png", "logo.png"):
        p = Path(__file__).parent / "static" / fname
        if p.exists():
            b64 = _b64.b64encode(p.read_bytes()).decode()
            logo_tag = (f'<img src="data:image/png;base64,{b64}" '
                        f'style="height:32px;object-fit:contain;display:block;" alt="Lover Clinic">')
            break
    if not logo_tag and _LOGO_SRC:
        logo_tag = (f'<img src="{_LOGO_SRC}" '
                    f'style="height:32px;object-fit:contain;display:block;'
                    f'filter:brightness(0) invert(1);" alt="Lover Clinic">')
    if not logo_tag:
        logo_tag = (
            '<span style="font-family:ui-sans-serif,system-ui,sans-serif;font-weight:900;'
            'font-size:20px;letter-spacing:2px;color:#fff;line-height:1;">'
            'L<span style="color:#dc2626;">O</span>VER'
            '<span style="font-size:11px;font-weight:600;color:#6b7280;'
            'letter-spacing:3px;margin-left:4px;">CLINIC</span>'
            '</span>'
        )

    # ── Icon ──────────────────────────────────────────────
    icon_tag = ""
    if _ICON_SRC:
        icon_tag = (f'<img src="{_ICON_SRC}" '
                    f'style="width:36px;height:36px;border-radius:10px;object-fit:cover;'
                    f'border:1px solid #222;flex-shrink:0;">')

    # ── Badge pills ───────────────────────────────────────
    badges = [
        ("🔬", "Upscale Photo"),
        ("🎬", "Upscale Video"),
        ("✂️", "Remove BG"),
        ("✨", "Enhance"),
        ("📐", "Resize"),
        ("🖼️", "Crop"),
        ("🔄", "Convert"),
    ]
    badge_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:4px 10px;border-radius:9999px;border:1px solid #222;'
        f'font-size:0.65rem;font-weight:600;color:#6b7280;letter-spacing:0.04em;'
        f'white-space:nowrap;">{ic} {lb}</span>'
        for ic, lb in badges
    )

    return (
        # ── Top bar ──────────────────────────────────────
        '<div style="background:#050505;border-bottom:1px solid #222;">'
        '<div style="max-width:1400px;margin:0 auto;padding:14px 24px;'
        'display:flex;align-items:center;justify-content:space-between;gap:16px;">'
        # left: icon + logo
        '<div style="display:flex;align-items:center;gap:10px;flex-shrink:0;">'
        + (icon_tag if icon_tag else '') +
        logo_tag +
        '</div>'
        # center: subtitle
        '<div style="flex:1;text-align:center;">'
        '<span style="font-size:0.65rem;font-weight:700;letter-spacing:0.2em;'
        'color:#374151;text-transform:uppercase;">AI Image &amp; Video Processing Suite</span>'
        '</div>'
        # right: version badge
        '<div style="flex-shrink:0;">'
        '<span style="font-size:0.6rem;font-weight:600;color:#374151;'
        'border:1px solid #222;border-radius:9999px;padding:3px 10px;letter-spacing:0.06em;">'
        'v2.0</span>'
        '</div>'
        '</div></div>'
        # ── Feature pills bar ────────────────────────────
        '<div style="background:#050505;border-bottom:1px solid #161616;">'
        '<div style="max-width:1400px;margin:0 auto;padding:8px 24px;'
        'display:flex;align-items:center;gap:6px;overflow-x:auto;'
        'scrollbar-width:none;-ms-overflow-style:none;">'
        + badge_html +
        '</div></div>'
    )


WARN_GPU = """<div class="info-box">
⚠️ Video upscaling is GPU-intensive. Processing may take several minutes per minute of video.
GPU strongly recommended.
</div>"""


# ============================================================
# GRADIO UI
# ============================================================

CROP_INIT_JS = """
() => {
  setTimeout(function() {
    var canvas = document.getElementById('lc-canvas');
    var img    = document.getElementById('lc-crop-img');
    if (!canvas || !img) return;
    var ctx   = canvas.getContext('2d');
    var origW = parseInt(img.getAttribute('data-w'));
    var origH = parseInt(img.getAttribute('data-h'));
    var maxW  = (canvas.parentElement ? canvas.parentElement.clientWidth : 600) - 24;
    var maxH  = Math.round(window.innerHeight * 0.46);
    if (!maxW || maxW < 10) maxW = 600;
    if (!maxH || maxH < 10) maxH = 460;
    var scale = Math.min(1, maxW / origW, maxH / origH);
    canvas.width  = Math.round(origW * scale);
    canvas.height = Math.round(origH * scale);

    // ── Zoom ──────────────────────────────────────────────
    var baseScale  = scale;
    var zoomFactor = 1.0;
    var zoomSteps  = [0.25,0.33,0.5,0.67,0.75,1.0,1.25,1.5,2.0,2.5,3.0];
    function applyZoom(newFactor){
      var oldW=canvas.width, oldH=canvas.height;
      zoomFactor=newFactor;
      var ns=baseScale*zoomFactor;
      var nW=Math.round(origW*ns), nH=Math.round(origH*ns);
      if(hasSel && oldW>0){
        var fx=nW/oldW, fy=nH/oldH;
        sx=Math.round(sx*fx); sy=Math.round(sy*fy);
        ex=Math.round(ex*fx); ey=Math.round(ey*fy);
      }
      canvas.width=nW; canvas.height=nH;
      redraw(); updateCoords();
      var zEl=document.getElementById('lc-zoom-val');
      if(zEl) zEl.textContent=Math.round(zoomFactor*100)+'%';
    }

    // ── State ─────────────────────────────────────────────
    var sx=0, sy=0, ex=0, ey=0;
    var hasSel = false;
    var isDown = false;
    var dragMode = 'draw';          // 'draw' | 'move' | 'resize-<id>'
    var dragStart = null;           // snapshot of sx/sy/ex/ey at mousedown
    var lockedRatio = null;         // null = free  |  {w, h}
    var showGrid = false;
    var HR = 7;                     // handle radius px

    // ── Helpers ───────────────────────────────────────────
    function getPos(e) {
      var r = canvas.getBoundingClientRect();
      var sc = canvas.width / (canvas.clientWidth || canvas.width);
      return { x: Math.round((e.clientX-r.left)*sc),
               y: Math.round((e.clientY-r.top)*sc) };
    }
    function clamp(v,lo,hi){ return Math.max(lo, Math.min(hi, v)); }
    function clampAll() {
      sx=clamp(sx,0,canvas.width); sy=clamp(sy,0,canvas.height);
      ex=clamp(ex,0,canvas.width); ey=clamp(ey,0,canvas.height);
    }
    function selRect() {
      return { x:Math.min(sx,ex), y:Math.min(sy,ey),
               w:Math.abs(ex-sx), h:Math.abs(ey-sy) };
    }
    function handles() {
      var r=selRect(); if(r.w<4||r.h<4) return [];
      var mx=r.x+r.w/2, my=r.y+r.h/2;
      return [
        {id:'nw',x:r.x,       y:r.y},
        {id:'n', x:mx,         y:r.y},
        {id:'ne',x:r.x+r.w,   y:r.y},
        {id:'e', x:r.x+r.w,   y:my},
        {id:'se',x:r.x+r.w,   y:r.y+r.h},
        {id:'s', x:mx,         y:r.y+r.h},
        {id:'sw',x:r.x,        y:r.y+r.h},
        {id:'w', x:r.x,        y:my},
      ];
    }
    function hitHandle(p) {
      var hs=handles();
      for(var i=0;i<hs.length;i++){
        if(Math.abs(p.x-hs[i].x)<=HR+2 && Math.abs(p.y-hs[i].y)<=HR+2) return hs[i].id;
      }
      return null;
    }
    function insideSel(p) {
      var r=selRect();
      return r.w>4&&r.h>4 && p.x>r.x+HR && p.x<r.x+r.w-HR && p.y>r.y+HR && p.y<r.y+r.h-HR;
    }
    function applyRatioH(w) {
      return lockedRatio ? Math.round(w * lockedRatio.h / lockedRatio.w) : null;
    }

    // ── Drawing ───────────────────────────────────────────
    function redraw() {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img,0,0,canvas.width,canvas.height);
      var r=selRect();
      if((!hasSel && !isDown) || r.w<2 || r.h<2) return;
      // dim outside
      ctx.fillStyle='rgba(0,0,0,0.5)';
      ctx.fillRect(0,0,canvas.width,r.y);
      ctx.fillRect(0,r.y+r.h,canvas.width,canvas.height-r.y-r.h);
      ctx.fillRect(0,r.y,r.x,r.h);
      ctx.fillRect(r.x+r.w,r.y,canvas.width-r.x-r.w,r.h);
      ctx.save();
      // selection border
      ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.setLineDash([]);
      ctx.strokeRect(r.x+0.5,r.y+0.5,r.w-1,r.h-1);
      // rule-of-thirds grid
      if(showGrid) {
        ctx.strokeStyle='rgba(255,255,255,0.55)'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        var gx1=r.x+r.w/3, gx2=r.x+2*r.w/3, gy1=r.y+r.h/3, gy2=r.y+2*r.h/3;
        ctx.beginPath();
        ctx.moveTo(gx1,r.y); ctx.lineTo(gx1,r.y+r.h);
        ctx.moveTo(gx2,r.y); ctx.lineTo(gx2,r.y+r.h);
        ctx.moveTo(r.x,gy1); ctx.lineTo(r.x+r.w,gy1);
        ctx.moveTo(r.x,gy2); ctx.lineTo(r.x+r.w,gy2);
        ctx.stroke();
      }
      // handles
      ctx.setLineDash([]);
      var hs=handles();
      for(var i=0;i<hs.length;i++){
        ctx.beginPath(); ctx.arc(hs[i].x,hs[i].y,HR,0,2*Math.PI);
        ctx.fillStyle='#fff'; ctx.fill();
        ctx.strokeStyle='#1d4ed8'; ctx.lineWidth=1.5; ctx.stroke();
      }
      ctx.restore();
    }

    // ── Coords Update ─────────────────────────────────────
    function updateCoords() {
      var scX=origW/canvas.width, scY=origH/canvas.height;
      var x0=Math.max(0,Math.round(Math.min(sx,ex)*scX));
      var y0=Math.max(0,Math.round(Math.min(sy,ey)*scY));
      var x1=Math.min(origW,Math.round(Math.max(sx,ex)*scX));
      var y1=Math.min(origH,Math.round(Math.max(sy,ey)*scY));
      var coords=x0+','+y0+','+x1+','+y1;
      window._lcCropCoords = coords;
      var el=document.querySelector('#lc-crop-coords textarea');
      if(!el) el=document.querySelector('#lc-crop-coords input');
      if(el){ try{
        var p=el.tagName==='TEXTAREA'?window.HTMLTextAreaElement.prototype:window.HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(p,'value').set.call(el,coords);
        el.dispatchEvent(new Event('input',{bubbles:true}));
        el.dispatchEvent(new Event('change',{bubbles:true}));
      }catch(e){} }
      var rStr=lockedRatio?(lockedRatio.w+':'+lockedRatio.h):'อิสระ';
      var info=document.getElementById('lc-crop-info');
      if(info) info.textContent=(x1-x0)+'×'+(y1-y0)+' px  ·  ('+x0+', '+y0+')  ·  ratio: '+rStr+'  ·  ต้นฉบับ '+origW+'×'+origH+' px';
    }

    // ── Cursor ────────────────────────────────────────────
    var CURSORS={'nw':'nw-resize','n':'n-resize','ne':'ne-resize','e':'e-resize',
                 'se':'se-resize','s':'s-resize','sw':'sw-resize','w':'w-resize'};

    // ── Mouse ─────────────────────────────────────────────
    canvas.onmousedown = function(e) {
      var p=getPos(e); e.preventDefault();
      var h=hitHandle(p);
      if(h){ dragMode='resize-'+h; dragStart={x:p.x,y:p.y,sx:sx,sy:sy,ex:ex,ey:ey}; }
      else if(hasSel && insideSel(p)){ dragMode='move'; dragStart={x:p.x,y:p.y,sx:sx,sy:sy,ex:ex,ey:ey}; }
      else { dragMode='draw'; sx=p.x; sy=p.y; ex=p.x; ey=p.y; hasSel=false; }
      isDown=true;
    };

    canvas.onmousemove = function(e) {
      var p=getPos(e);
      if(!isDown) {
        var h=hitHandle(p);
        canvas.style.cursor = h ? (CURSORS[h]||'pointer') : (hasSel&&insideSel(p)?'move':'crosshair');
        return;
      }
      if(dragMode==='draw') {
        ex=clamp(p.x,0,canvas.width); ey=clamp(p.y,0,canvas.height);
        if(lockedRatio){ var newH=applyRatioH(Math.abs(ex-sx)); ey=sy+newH*(p.y>=sy?1:-1); ey=clamp(ey,0,canvas.height); }
      } else if(dragMode==='move') {
        var dx=p.x-dragStart.x, dy=p.y-dragStart.y;
        var w=dragStart.ex-dragStart.sx, h=dragStart.ey-dragStart.sy;
        sx=clamp(dragStart.sx+dx, 0, canvas.width-Math.abs(w));
        sy=clamp(dragStart.sy+dy, 0, canvas.height-Math.abs(h));
        ex=sx+w; ey=sy+h;
      } else if(dragMode.startsWith('resize-')) {
        var hid=dragMode.slice(7);
        var nx=clamp(p.x,0,canvas.width), ny=clamp(p.y,0,canvas.height);
        if(hid==='se'||hid==='e'||hid==='ne') ex=nx;
        if(hid==='sw'||hid==='w'||hid==='nw') sx=nx;
        if(hid==='se'||hid==='s'||hid==='sw') ey=ny;
        if(hid==='ne'||hid==='n'||hid==='nw') sy=ny;
        if(lockedRatio) {
          var cw=Math.abs(ex-sx);
          var ch=applyRatioH(cw);
          if(hid==='n'||hid==='nw'||hid==='ne') sy=ey-(ch*(ey>dragStart.sy?1:-1));
          else ey=sy+ch*(ey>=dragStart.ey?1:-1);
          clampAll();
        }
      }
      redraw();
    };

    canvas.onmouseup = function(e) {
      isDown=false; var p=getPos(e);
      if(dragMode==='draw'){
        ex=clamp(p.x,0,canvas.width); ey=clamp(p.y,0,canvas.height);
        if(lockedRatio){ var newH=applyRatioH(Math.abs(ex-sx)); ey=sy+newH*(p.y>=sy?1:-1); ey=clamp(ey,0,canvas.height); }
      }
      hasSel=(Math.abs(ex-sx)>4 && Math.abs(ey-sy)>4);
      dragMode='draw'; redraw(); if(hasSel) updateCoords();
    };

    canvas.onmouseleave = function(e) {
      if(isDown && dragMode==='draw'){ isDown=false; hasSel=(Math.abs(ex-sx)>4&&Math.abs(ey-sy)>4); redraw(); if(hasSel) updateCoords(); dragMode='draw'; }
    };

    // ── Toolbar Buttons ───────────────────────────────────
    function setRatio(rStr) {
      if(rStr==='free'){ lockedRatio=null; return; }
      var p=rStr.split(':'); lockedRatio={w:parseFloat(p[0]),h:parseFloat(p[1])};
      if(hasSel){
        var w=Math.abs(ex-sx), newH=applyRatioH(w);
        ey=sy+newH*(ey>=sy?1:-1); ey=clamp(ey,0,canvas.height);
        redraw(); updateCoords();
      }
    }
    function centerAndApplyRatio(rw,rh) {
      lockedRatio={w:rw,h:rh};
      var tr=rw/rh, cr=canvas.width/canvas.height;
      var sw,sh;
      if(tr>cr){ sw=Math.round(canvas.width*0.92); sh=Math.round(sw/tr); }
      else     { sh=Math.round(canvas.height*0.92); sw=Math.round(sh*tr); }
      sx=Math.round((canvas.width-sw)/2); sy=Math.round((canvas.height-sh)/2);
      ex=sx+sw; ey=sy+sh; hasSel=true; redraw(); updateCoords();
    }

    // Ratio buttons
    document.querySelectorAll('.lc-ratio-btn').forEach(function(btn){
      btn.addEventListener('click',function(){
        document.querySelectorAll('.lc-ratio-btn').forEach(function(b){b.classList.remove('lc-active');});
        document.querySelectorAll('.lc-chip').forEach(function(b){b.classList.remove('lc-active');});
        btn.classList.add('lc-active');
        setRatio(btn.getAttribute('data-ratio'));
      });
    });

    // Social presets
    document.querySelectorAll('.lc-chip').forEach(function(btn){
      btn.addEventListener('click',function(){
        document.querySelectorAll('.lc-ratio-btn').forEach(function(b){b.classList.remove('lc-active');});
        document.querySelectorAll('.lc-chip').forEach(function(b){b.classList.remove('lc-active');});
        btn.classList.add('lc-active');
        var p=btn.getAttribute('data-ratio').split(':');
        centerAndApplyRatio(parseFloat(p[0]),parseFloat(p[1]));
      });
    });

    // Grid toggle
    var gBtn=document.getElementById('lc-btn-grid');
    if(gBtn) gBtn.addEventListener('click',function(){ showGrid=!showGrid; gBtn.classList.toggle('lc-active',showGrid); redraw(); });

    // Center
    var cBtn=document.getElementById('lc-btn-center');
    if(cBtn) cBtn.addEventListener('click',function(){
      if(!hasSel) return;
      var w=Math.abs(ex-sx), h=Math.abs(ey-sy);
      sx=Math.round((canvas.width-w)/2); sy=Math.round((canvas.height-h)/2);
      ex=sx+w; ey=sy+h; redraw(); updateCoords();
    });

    // Swap portrait/landscape
    var sBtn=document.getElementById('lc-btn-swap');
    if(sBtn) sBtn.addEventListener('click',function(){
      if(lockedRatio) lockedRatio={w:lockedRatio.h,h:lockedRatio.w};
      if(hasSel){
        var cx=(sx+ex)/2, cy=(sy+ey)/2;
        var w=Math.abs(ex-sx), h=Math.abs(ey-sy);
        sx=Math.round(cx-h/2); ex=Math.round(cx+h/2);
        sy=Math.round(cy-w/2); ey=Math.round(cy+w/2);
        clampAll(); redraw(); updateCoords();
      }
    });

    // Reset
    var rBtn=document.getElementById('lc-btn-reset');
    if(rBtn) rBtn.addEventListener('click',function(){
      sx=0;sy=0;ex=0;ey=0;hasSel=false;window._lcCropCoords='';
      redraw();
      var info=document.getElementById('lc-crop-info');
      if(info) info.textContent='ลากเพื่อเลือกพื้นที่ · ต้นฉบับ '+origW+'×'+origH+' px';
    });

    // Zoom In / Out / Fit
    function nearestZoomIdx(f){
      var best=0, bd=Math.abs(zoomSteps[0]-f);
      for(var i=1;i<zoomSteps.length;i++){var d=Math.abs(zoomSteps[i]-f);if(d<bd){bd=d;best=i;}}
      return best;
    }
    var ziBtn=document.getElementById('lc-btn-zoom-in');
    if(ziBtn) ziBtn.addEventListener('click',function(){
      var idx=nearestZoomIdx(zoomFactor);
      applyZoom(zoomSteps[Math.min(zoomSteps.length-1,idx+1)]);
    });
    var zoBtn=document.getElementById('lc-btn-zoom-out');
    if(zoBtn) zoBtn.addEventListener('click',function(){
      var idx=nearestZoomIdx(zoomFactor);
      applyZoom(zoomSteps[Math.max(0,idx-1)]);
    });
    var zfBtn=document.getElementById('lc-btn-zoom-fit');
    if(zfBtn) zfBtn.addEventListener('click',function(){ applyZoom(1.0); });

    // ── Init ──────────────────────────────────────────────
    function initDraw() {
      redraw();
      var info=document.getElementById('lc-crop-info');
      if(info) info.textContent='ลากเพื่อเลือกพื้นที่ · ต้นฉบับ '+origW+'×'+origH+' px';
    }
    if(img.complete){ initDraw(); } else { img.onload=initDraw; }
  }, 150);
}
"""


def build_app():
    cfg = _load_settings()          # ← load persisted settings once at startup

    with gr.Blocks(title="Lover Clinic AI Video Tools", css=CSS) as demo:
        gr.HTML(_header_html())

        with gr.Tabs():

            # ── AI: Photo Upscale ──────────────────────────
            with gr.Tab("🔬 AI Upscale Photo"):
                gr.HTML('<div class="sec-head">AI Photo Upscaler · Real-ESRGAN</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        up_in = gr.Image(label="Input Image", type="pil", height=380)
                        with gr.Row():
                            up_scale = gr.Radio(
                                choices=[2, 4], value=cfg["up_scale"], label="Scale Factor", type="value"
                            )
                            up_model = gr.Radio(
                                choices=["General Photo", "Anime / Illustration"],
                                value=cfg["up_model"], label="Model"
                            )
                        up_btn = gr.Button("🚀 Upscale Photo", variant="primary")
                    with gr.Column(scale=1):
                        up_out = gr.Image(label="Upscaled Result", height=380, interactive=False)
                        up_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    up_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["up_fmt"], label="Save Format", scale=1)
                up_out_dir, _ = _save_dir_row("photo")
                up_out_dir.value = cfg["up_out_dir"]
                up_btn.click(upscale_photo, inputs=[up_in, up_scale, up_model, up_out_dir, up_fmt], outputs=[up_out, up_status])
                # persist on change
                up_scale.change(_make_saver("up_scale"), inputs=[up_scale])
                up_model.change(_make_saver("up_model"), inputs=[up_model])
                up_fmt.change(_make_saver("up_fmt"), inputs=[up_fmt])
                up_out_dir.change(_make_saver("up_out_dir"), inputs=[up_out_dir])

            # ── AI: Video Upscale ──────────────────────────
            with gr.Tab("🎬 AI Upscale Video"):
                gr.HTML('<div class="sec-head">AI Video Upscaler · Real-ESRGAN</div>')
                gr.HTML(WARN_GPU)
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Input Video")
                        with gr.Row():
                            vid_scale = gr.Radio(
                                choices=[2, 4], value=cfg["vid_scale"], label="Scale Factor", type="value"
                            )
                            vid_model = gr.Radio(
                                choices=["General", "Anime"],
                                value=cfg["vid_model"], label="Model"
                            )
                        vid_btn = gr.Button("🚀 Upscale Video", variant="primary")
                    with gr.Column(scale=1):
                        vid_out = gr.Video(label="Upscaled Video", visible=True)
                        vid_status = gr.Markdown(value="", elem_classes=["lc-status"])
                vid_out_dir, _ = _save_dir_row("video")
                vid_out_dir.value = cfg["vid_out_dir"]
                vid_btn.click(upscale_video, inputs=[vid_in, vid_scale, vid_model, vid_out_dir], outputs=[vid_out, vid_status])
                vid_scale.change(_make_saver("vid_scale"), inputs=[vid_scale])
                vid_model.change(_make_saver("vid_model"), inputs=[vid_model])
                vid_out_dir.change(_make_saver("vid_out_dir"), inputs=[vid_out_dir])

            # ── AI: Remove Background ──────────────────────
            with gr.Tab("✂️ AI Remove BG"):
                gr.HTML('<div class="sec-head">AI Background Remover · BiRefNet</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        bg_in = gr.Image(label="Input Image", type="pil", height=340)
                        bg_model = gr.Dropdown(
                            choices=["BiRefNet — General (Best)", "BiRefNet — Portrait", "U2Net", "RMBG 1.4"],
                            value=cfg["bg_model"], label="AI Model",
                        )
                        bg_option = gr.Radio(
                            choices=["Transparent", "White", "Black", "Red", "Custom Image"],
                            value=cfg["bg_option"], label="Background Replacement",
                        )
                        bg_custom = gr.Image(
                            label="Custom Background Image", type="pil",
                            visible=(cfg["bg_option"] == "Custom Image"), height=130
                        )
                        bg_btn = gr.Button("🚀 Remove Background", variant="primary")

                        def toggle_custom(choice):
                            return gr.update(visible=(choice == "Custom Image"))
                        bg_option.change(toggle_custom, bg_option, bg_custom)

                    with gr.Column(scale=1):
                        bg_out = gr.Image(label="Result", height=340, interactive=False)
                        bg_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    bg_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["bg_fmt"], label="Save Format (Transparent → PNG always)", scale=1)
                bg_out_dir, _ = _save_dir_row("remove_bg")
                bg_out_dir.value = cfg["bg_out_dir"]
                bg_btn.click(remove_background, inputs=[bg_in, bg_model, bg_option, bg_custom, bg_out_dir, bg_fmt], outputs=[bg_out, bg_status])
                bg_model.change(_make_saver("bg_model"), inputs=[bg_model])
                bg_option.change(_make_saver("bg_option"), inputs=[bg_option])
                bg_fmt.change(_make_saver("bg_fmt"), inputs=[bg_fmt])
                bg_out_dir.change(_make_saver("bg_out_dir"), inputs=[bg_out_dir])

            # ── AI: Enhance Image ──────────────────────────
            with gr.Tab("✨ AI Enhance Image"):
                gr.HTML('<div class="sec-head">AI Image Enhancer · GFPGAN</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        enh_in = gr.Image(label="Input Image", type="pil", height=340)
                        enh_scale = gr.Radio(
                            choices=[1, 2], value=cfg["enh_scale"], label="Upscale Factor", type="value"
                        )
                        enh_bg = gr.Checkbox(value=cfg["enh_bg"], label="Also enhance background (Real-ESRGAN)")
                        enh_btn = gr.Button("🚀 Enhance Image", variant="primary")
                    with gr.Column(scale=1):
                        enh_out = gr.Image(label="Enhanced Result", height=340, interactive=False)
                        enh_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    enh_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["enh_fmt"], label="Save Format", scale=1)
                enh_out_dir, _ = _save_dir_row("enhance")
                enh_out_dir.value = cfg["enh_out_dir"]
                enh_btn.click(enhance_image, inputs=[enh_in, enh_scale, enh_bg, enh_out_dir, enh_fmt], outputs=[enh_out, enh_status])
                enh_scale.change(_make_saver("enh_scale"), inputs=[enh_scale])
                enh_bg.change(_make_saver("enh_bg"), inputs=[enh_bg])
                enh_fmt.change(_make_saver("enh_fmt"), inputs=[enh_fmt])
                enh_out_dir.change(_make_saver("enh_out_dir"), inputs=[enh_out_dir])

            # ── TOOLS: Resize ──────────────────────────────
            with gr.Tab("📐 Resize Image"):
                gr.HTML('<div class="sec-head">Image Resizer</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        res_in = gr.Image(label="Input Image", type="pil", height=320)
                        res_dims = gr.Textbox(label="Current Dimensions", interactive=False, lines=1)

                        def show_dims(img):
                            return f"{img.width} × {img.height} px" if img else ""
                        res_in.change(show_dims, res_in, res_dims)

                        with gr.Row():
                            res_w = gr.Number(label="Width (px)", value=cfg["res_w"], minimum=1, maximum=16384)
                            res_h = gr.Number(label="Height (px)", value=cfg["res_h"], minimum=1, maximum=16384)
                        res_ar = gr.Checkbox(value=cfg["res_ar"], label="Maintain Aspect Ratio")
                        res_filter = gr.Dropdown(
                            choices=["Lanczos (Best Quality)", "Bicubic", "Bilinear", "Nearest (Fastest)"],
                            value=cfg["res_filter"], label="Resample Filter"
                        )
                        res_btn = gr.Button("📐 Resize", variant="primary")
                    with gr.Column(scale=1):
                        res_out = gr.Image(label="Resized Image", height=320, interactive=False)
                        res_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    res_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["res_fmt"], label="Save Format", scale=1)
                res_out_dir, _ = _save_dir_row("resize")
                res_out_dir.value = cfg["res_out_dir"]
                res_btn.click(resize_image, inputs=[res_in, res_w, res_h, res_ar, res_filter, res_out_dir, res_fmt], outputs=[res_out, res_status])
                res_w.change(_make_saver("res_w"), inputs=[res_w])
                res_h.change(_make_saver("res_h"), inputs=[res_h])
                res_ar.change(_make_saver("res_ar"), inputs=[res_ar])
                res_filter.change(_make_saver("res_filter"), inputs=[res_filter])
                res_fmt.change(_make_saver("res_fmt"), inputs=[res_fmt])
                res_out_dir.change(_make_saver("res_out_dir"), inputs=[res_out_dir])

            # ── TOOLS: Crop ────────────────────────────────
            with gr.Tab("✂️ Crop Image"):
                gr.HTML('<div class="sec-head">Image Cropper · ลากเพื่อเลือกพื้นที่</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        crop_in = gr.Image(label="อัปโหลดรูปภาพ", type="pil", height=180)
                        crop_display = gr.HTML(
                            '<div style="color:#666;text-align:center;padding:32px;">⬆️ อัปโหลดรูปภาพก่อน</div>'
                        )
                        crop_coords = gr.Textbox(
                            value="", visible=False, elem_id="lc-crop-coords", label="coords"
                        )
                        with gr.Row():
                            crop_btn = gr.Button("✂️ Crop", variant="primary", scale=2)
                    with gr.Column(scale=1):
                        crop_out = gr.Image(label="Cropped Result", interactive=False)
                        crop_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    crop_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["crop_fmt"], label="Save Format", scale=1)
                crop_out_dir, _ = _save_dir_row("crop")
                crop_out_dir.value = cfg["crop_out_dir"]
                crop_in.change(_make_cropper_html, inputs=[crop_in], outputs=[crop_display]).then(fn=None, js=CROP_INIT_JS)
                crop_btn.click(
                    crop_image,
                    inputs=[crop_in, crop_coords, crop_out_dir, crop_fmt],
                    outputs=[crop_out, crop_status],
                    js="(img, coords, dir, fmt) => [img, (window._lcCropCoords && window._lcCropCoords.indexOf(',')>0 ? window._lcCropCoords : coords), dir, fmt]",
                )
                crop_fmt.change(_make_saver("crop_fmt"), inputs=[crop_fmt])
                crop_out_dir.change(_make_saver("crop_out_dir"), inputs=[crop_out_dir])

            # ── TOOLS: Convert ─────────────────────────────
            with gr.Tab("🔄 Convert Format"):
                gr.HTML('<div class="sec-head">Image Format Converter</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        conv_in = gr.Image(label="Input Image", type="pil", height=300)
                        conv_fmt = gr.Radio(
                            choices=["JPEG", "PNG", "WEBP", "BMP", "TIFF"],
                            value=cfg["conv_fmt"], label="Output Format"
                        )
                        conv_q = gr.Slider(1, 100, value=cfg["conv_q"], step=1, label="Quality (JPEG / WEBP only)")
                        conv_btn = gr.Button("🔄 Convert", variant="primary")
                    with gr.Column(scale=1):
                        conv_preview = gr.Image(label="Preview", height=300, interactive=False)
                        conv_file = gr.File(label="⬇️ Download Converted File")
                        conv_status = gr.Markdown(value="", elem_classes=["lc-status"])
                conv_out_dir, _ = _save_dir_row("convert")
                conv_out_dir.value = cfg["conv_out_dir"]
                conv_btn.click(convert_format, inputs=[conv_in, conv_fmt, conv_q, conv_out_dir], outputs=[conv_preview, conv_file, conv_status])
                conv_fmt.change(_make_saver("conv_fmt"), inputs=[conv_fmt])
                conv_q.change(_make_saver("conv_q"), inputs=[conv_q])
                conv_out_dir.change(_make_saver("conv_out_dir"), inputs=[conv_out_dir])

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    static_dir = str(Path(__file__).parent / "static")
    favicon = str(Path(__file__).parent / "static" / "icon.png")

    app = build_app()
    logger.info(f"Launching on http://127.0.0.1:{args.port}")
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=True,
        favicon_path=favicon,
        allowed_paths=[static_dir, str(_OUTPUT_ROOT)],
    )
