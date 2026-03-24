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
    level=logging.INFO,
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
    "vid_scale":   4,
    "vid_model":   "General (Best Quality)",
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
    "res_filter":  "Lanczos (คุณภาพสูงสุด)",
    "res_fmt":     "PNG",
    "res_out_dir": None,
    # Crop
    "crop_fmt":     "PNG",
    "crop_out_dir": None,
    # Convert
    "conv_fmt":     "WEBP",
    "conv_q":       95,
    "conv_out_dir": None,
    # Download
    "dl_out_dir":   None,
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

    # ── Migrate renamed model labels (old saved values → new names) ──────────
    _vid_model_migration = {
        "General": "General (Best Quality)",   # old → new
        "Anime":   "Anime / Cartoon",
    }
    _valid_vid_models = {"General (Best Quality)", "General (Fast)", "Anime / Cartoon"}
    if cfg.get("vid_model") not in _valid_vid_models:
        cfg["vid_model"] = _vid_model_migration.get(
            cfg.get("vid_model"), "General (Best Quality)"
        )

    _valid_up_models = {"General Photo", "General (Lightweight)", "Anime / Illustration"}
    if cfg.get("up_model") not in _valid_up_models:
        cfg["up_model"] = "General Photo"

    # fill None dirs with real defaults
    _dir_defaults = {
        "up_out_dir":   "photo",
        "vid_out_dir":  "video",
        "bg_out_dir":   "remove_bg",
        "enh_out_dir":  "enhance",
        "res_out_dir":  "resize",
        "crop_out_dir": "crop",
        "conv_out_dir": "convert",
        "dl_out_dir":   "download",
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
# STARTUP DEVICE HEALTH CHECK
# ============================================================

def _startup_device_check():
    """Run once at startup to log device status and confirm GPU availability."""
    try:
        import torch
        ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            logger.info(f"✅ GPU ready  — {gpu}  |  VRAM {vram} MiB  |  torch {ver}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(f"✅ GPU ready  — Apple Silicon MPS  |  torch {ver}")
        else:
            if "+cpu" in ver:
                logger.warning(
                    f"⚠️  CPU-only torch detected ({ver}). "
                    f"GPU acceleration NOT available. "
                    f"If you have an NVIDIA/AMD GPU, run Fix → Re-install to restore GPU support."
                )
            else:
                logger.warning(
                    f"⚠️  CUDA unavailable (torch {ver}). "
                    f"CUDA DLLs may not be loaded. Processing will use CPU."
                )
    except Exception as e:
        logger.warning(f"⚠️  torch unavailable at startup: {e}. All processing will use CPU/PIL fallback.")

_startup_device_check()


def _cleanup_old_logs(max_age_hours: float = 1.0):
    """Delete log files in app/logs/ that were last modified more than max_age_hours ago."""
    import time
    cutoff = time.time() - max_age_hours * 3600
    for log_file in _log_dir.glob("*.log"):
        try:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
        except Exception:
            pass  # Never crash on log cleanup

_cleanup_old_logs(max_age_hours=1.0)

# ============================================================
# LAZY MODEL MANAGERS
# ============================================================

_realesrgan_cache = {}
_gfpgan_cache = {}
_rembg_sessions = {}


_device_cache: dict = {}

def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu'. Result is cached after first call."""
    if "dev" in _device_cache:
        return _device_cache["dev"]
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    except (ImportError, OSError):
        dev = "cpu"
    _device_cache["dev"] = dev
    return dev


def get_realesrgan(scale=4, model_type="general"):
    key = f"{scale}_{model_type}"
    # Include device in cache key — forces re-load if device changes between calls
    device = get_device()
    full_key = f"{key}_{device}"
    if full_key not in _realesrgan_cache:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        logger.info(f"Loading RealESRGAN model: {model_type} x{scale} on {device}")

        if model_type == "anime":
            # realesr-animevideov3 — SRVGGNetCompact (8 MB)
            # Best temporal stability for ANIME VIDEO — low flickering
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_conv=16, upscale=4, act_type='prelu')
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.5.0/realesr-animevideov3.pth")
            tile = 400
            out_scale = 4

        elif model_type == "anime-still":
            # RealESRGAN_x4plus_anime_6B — RRDBNet 6 blocks (17 MB)
            # Best quality for ANIME / ILLUSTRATION STILL IMAGES
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
            tile = 400
            out_scale = 4

        elif model_type == "general-fast":
            # realesr-general-x4v3 — SRVGGNetCompact (16 MB)
            # Lightweight general model — fast, good temporal stability for VIDEO
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_conv=32, upscale=4, act_type='prelu')
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.5.0/realesr-general-x4v3.pth")
            tile = 512
            out_scale = 4

        elif scale == 2:
            # RealESRGAN_x2plus — RRDBNet 23 blocks (64 MB) — general 2x
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.1/RealESRGAN_x2plus.pth")
            tile = 512
            out_scale = 2

        else:
            # RealESRGAN_x4plus — RRDBNet 23 blocks (64 MB) — flagship general 4x
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            model_url = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.1.0/RealESRGAN_x4plus.pth")
            tile = 512
            out_scale = 4

        _realesrgan_cache[full_key] = RealESRGANer(
            scale=out_scale,
            model_path=model_url,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda"),
            device=device,
        )
    return _realesrgan_cache[full_key]


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
        _photo_model_map = {
            "General Photo":          "general",
            "General (Lightweight)":  "general-fast",
            "Anime / Illustration":   "anime-still",
        }
        mt = _photo_model_map.get(model_type, "general")
        # anime-still and general-fast are 4x-only models
        actual_scale = 4 if mt in ("anime-still", "general-fast") else int(scale)
        logger.info(f"upscale_photo: model={mt} scale={actual_scale}x device={get_device()}")
        upsampler = get_realesrgan(scale=actual_scale, model_type=mt)

        progress(0.3, desc="Upscaling…")
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(img_cv2, outscale=actual_scale)

        progress(0.95, desc="Finalising…")
        result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        saved = _save_image(result, output_dir, "upscaled_photo", fmt=fmt)
        return result, f"✅ Upscaled {actual_scale}x — {result.width}×{result.height} px\n💾 {saved}"
    except Exception as e:
        logger.error(f"upscale_photo failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def upscale_video(video_path, scale, model_type, output_dir, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️ Please upload a video first."
    try:
        progress(0.05, desc="Loading model…")
        _video_model_map = {
            "General (Best Quality)": "general",
            "General (Fast)":         "general-fast",
            "Anime / Cartoon":        "anime",
        }
        mt = _video_model_map.get(model_type, "general")
        # general-fast and anime are 4x-only models
        actual_scale = 4 if mt in ("anime", "general-fast") else int(scale)
        logger.info(f"upscale_video: model={mt} scale={actual_scale}x device={get_device()}")
        upsampler = get_realesrgan(scale=actual_scale, model_type=mt)

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
            out_frame, _ = upsampler.enhance(frame_bgr, outscale=actual_scale)
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
        return gr.update(value=final_path, visible=True), f"✅ Video upscaled {actual_scale}x — saved to {final_path}"
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

        device = get_device()
        import torch
        logger.info(f"Loading GFPGAN model on {device} (upscale={upscale_factor})")
        restorer = GFPGANer(
            model_path=("https://github.com/TencentARC/GFPGAN/releases/"
                        "download/v1.3.4/GFPGANv1.4.pth"),
            upscale=int(upscale_factor),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device=torch.device(device),
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
  /* ── Reset: override Gradio's button globals ── */
  #lc-editor button {{
    -webkit-appearance:none!important;appearance:none!important;
    box-sizing:border-box!important;margin:0!important;
    line-height:1.2!important;outline:none!important;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif!important;
  }}
  #lc-editor {{
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    color:#e2e8f0;user-select:none;
  }}
  /* ── Canvas ─────────────────────────────────── */
  #lc-crop-wrap {{
    background:#050505;border:1px solid #1c1c1c;border-radius:10px;
    padding:6px;display:flex;justify-content:center;align-items:center;overflow:auto;
    min-height:160px;flex-shrink:0;box-sizing:border-box;
  }}
  #lc-canvas {{display:block;cursor:crosshair;border-radius:3px;}}
  /* ── Info bar ────────────────────────────────── */
  #lc-infobar {{
    display:flex;align-items:center;justify-content:space-between;gap:8px;
    background:#0a0a0a;border:1px solid #181818;border-radius:7px;
    padding:3px 10px;margin-top:4px;overflow:hidden;
  }}
  #lc-crop-info {{
    font-size:10px;font-variant-numeric:tabular-nums;color:#4b5563;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  }}
  #lc-hint {{font-size:9px;color:#2d3340;white-space:nowrap;flex-shrink:0;}}
  /* ── Toolbar ─────────────────────────────────── */
  #lc-toolbar {{
    background:#090909;border:1px solid #1c1c1c;border-radius:10px;
    margin-top:5px;overflow:hidden;
  }}
  /* ── Section row ────────────────────────────── */
  .lc-section {{display:flex;align-items:flex-start;border-bottom:1px solid #111;}}
  .lc-section:last-child {{border-bottom:none;}}
  .lc-section-lbl {{
    display:flex;align-items:center;justify-content:flex-end;
    font-size:clamp(6px,1.5vw,7.5px);font-weight:800;letter-spacing:.12em;
    text-transform:uppercase;color:#252d3a;
    width:clamp(32px,7vw,46px);min-width:clamp(32px,7vw,46px);
    padding:0 clamp(4px,1vw,7px);flex-shrink:0;
    border-right:1px solid #111;background:#070707;
  }}
  /* Content rows wrap — buttons never overflow the container */
  .lc-content {{
    display:flex;align-items:center;
    gap:clamp(3px,0.7vw,5px);
    padding:clamp(4px,1vw,6px) clamp(5px,1.2vw,8px);
    flex:1;min-width:0;flex-wrap:wrap;overflow:visible;
    row-gap:clamp(3px,0.7vw,5px);
  }}
  /* ── Pill group ─────────────────────────────── */
  .lc-grp {{
    display:flex;background:#0d0d0d;border:1px solid #1c1c1c;
    border-radius:6px;overflow:hidden;flex-shrink:0;
  }}
  /* ── ALL pill / ratio buttons — high-specificity + !important ── */
  #lc-editor .lc-btn-pill,
  #lc-editor .lc-ratio-btn {{
    background:transparent!important;color:#3a414f!important;
    border:none!important;border-right:1px solid #181818!important;
    padding:clamp(3px,0.7vw,5px) clamp(5px,1.3vw,9px)!important;
    font-size:clamp(9px,2vw,11px)!important;font-weight:600!important;
    cursor:pointer!important;white-space:nowrap!important;flex-shrink:0!important;
    transition:background .14s,color .14s!important;
  }}
  #lc-editor .lc-btn-pill:last-child,
  #lc-editor .lc-ratio-btn:last-child {{border-right:none!important;}}
  #lc-editor .lc-btn-pill:hover,
  #lc-editor .lc-ratio-btn:hover {{background:#161616!important;color:#c9cdd4!important;}}
  #lc-editor .lc-btn-pill.lc-active,
  #lc-editor .lc-ratio-btn.lc-active {{background:#b91c1c!important;color:#fff!important;}}
  /* ── Standalone action buttons ──────────────── */
  #lc-editor .lc-btn {{
    background:#0f0f0f!important;color:#4b5563!important;
    border:1px solid #1c1c1c!important;border-radius:6px!important;
    padding:clamp(3px,0.7vw,5px) clamp(5px,1.3vw,9px)!important;
    font-size:clamp(9px,2vw,11px)!important;font-weight:600!important;
    cursor:pointer!important;white-space:nowrap!important;flex-shrink:0!important;
    transition:all .14s!important;
  }}
  #lc-editor .lc-btn:hover {{background:#1a1a1a!important;color:#c9cdd4!important;border-color:#272727!important;}}
  #lc-editor .lc-btn.lc-active {{background:#15803d!important;color:#fff!important;border-color:#166534!important;}}
  /* ── Zoom +/- (slightly wider click target) ─── */
  #lc-editor #lc-btn-zoom-out,
  #lc-editor #lc-btn-zoom-in {{
    padding:clamp(3px,0.7vw,5px) clamp(7px,1.5vw,11px)!important;
    font-size:clamp(11px,2.5vw,14px)!important;
  }}
  /* ── Vertical divider ────────────────────────── */
  .lc-vdiv {{
    width:1px;height:clamp(14px,3vw,18px);
    background:#1c1c1c;margin:0 clamp(2px,0.5vw,4px);flex-shrink:0;
  }}
  /* ── Zoom value label ────────────────────────── */
  #lc-zoom-val {{
    font-size:clamp(9px,2vw,11px)!important;font-weight:700;color:#3a414f;
    min-width:clamp(26px,5vw,34px);text-align:center;flex-shrink:0;
  }}
  /* ── Social chips ────────────────────────────── */
  #lc-editor .lc-chip {{
    display:inline-flex!important;align-items:center!important;gap:2px!important;
    background:#0a0a0a!important;color:#4b5563!important;
    border:1px solid #1c1c1c!important;border-radius:9999px!important;
    padding:clamp(2px,0.5vw,4px) clamp(5px,1.2vw,9px)!important;
    font-size:clamp(8.5px,1.8vw,10px)!important;font-weight:600!important;
    cursor:pointer!important;white-space:nowrap!important;flex-shrink:0!important;
    transition:all .14s!important;
  }}
  #lc-editor .lc-chip:hover {{background:#161616!important;color:#c9cdd4!important;border-color:#272727!important;}}
  #lc-editor .lc-chip.lc-active {{background:#b91c1c!important;color:#fff!important;border-color:#b91c1c!important;}}
</style>

<div id="lc-editor">

  <!-- Canvas -->
  <div id="lc-crop-wrap">
    <canvas id="lc-canvas"></canvas>
    <img id="lc-crop-img" src="data:image/png;base64,{b64}" data-w="{w}" data-h="{h}" style="display:none;">
  </div>

  <!-- Info bar -->
  <div id="lc-infobar">
    <div id="lc-crop-info">⏳ กำลังเตรียม...</div>
    <div id="lc-hint">ลากเพื่อเลือก · ลากขอบ · ลากกลาง</div>
  </div>

  <!-- ══ Toolbar ══ -->
  <div id="lc-toolbar">

    <!-- Row 1 · Aspect Ratio (scrolls horizontally) -->
    <div class="lc-section">
      <span class="lc-section-lbl">สัดส่วน</span>
      <div class="lc-content">
        <div class="lc-grp">
          <button class="lc-btn-pill lc-active" data-ratio="free">Free</button>
          <button class="lc-btn-pill" data-ratio="orig" title="ล็อคตามอัตราส่วนของภาพต้นฉบับ">Orig</button>
          <button class="lc-btn-pill" data-ratio="1:1">1:1</button>
          <button class="lc-btn-pill" data-ratio="4:3">4:3</button>
          <button class="lc-btn-pill" data-ratio="3:4">3:4</button>
          <button class="lc-btn-pill" data-ratio="16:9">16:9</button>
          <button class="lc-btn-pill" data-ratio="9:16">9:16</button>
          <button class="lc-btn-pill" data-ratio="4:5">4:5</button>
          <button class="lc-btn-pill" data-ratio="5:4">5:4</button>
          <button class="lc-btn-pill" data-ratio="3:2">3:2</button>
          <button class="lc-btn-pill" data-ratio="2:3">2:3</button>
        </div>
        <div class="lc-vdiv"></div>
        <button class="lc-btn" id="lc-btn-swap"   title="สลับ Portrait ↔ Landscape">⇄ Swap</button>
        <button class="lc-btn" id="lc-btn-center" title="จัดกึ่งกลาง">⊙ Center</button>
        <button class="lc-btn" id="lc-btn-reset"  title="ล้างการเลือก">✕ Clear</button>
      </div>
    </div>

    <!-- Row 2 · Tools (scrolls horizontally) -->
    <div class="lc-section">
      <span class="lc-section-lbl">เครื่องมือ</span>
      <div class="lc-content">
        <div class="lc-grp">
          <button class="lc-ratio-btn" id="lc-btn-rot-l"   title="หมุน 90° ทวนเข็ม">↺ 90°L</button>
          <button class="lc-ratio-btn" id="lc-btn-rot-r"   title="หมุน 90° ตามเข็ม">↻ 90°R</button>
          <button class="lc-ratio-btn" id="lc-btn-rot-180" title="หมุน 180°">↕ 180°</button>
        </div>
        <div class="lc-vdiv"></div>
        <button class="lc-btn" id="lc-btn-flip-h"      title="กระจกซ้าย-ขวา">↔ H</button>
        <button class="lc-btn" id="lc-btn-flip-v"      title="กระจกบน-ล่าง">↕ V</button>
        <button class="lc-btn" id="lc-btn-xform-reset" title="รีเซ็ต rotate/flip">⟲</button>
        <div class="lc-vdiv"></div>
        <button class="lc-btn" id="lc-btn-img-grid" title="Grid ทั้งภาพ">⊟ Grid ภาพ</button>
        <button class="lc-btn" id="lc-btn-grid"     title="Grid ในกรอบ Crop">⊞ Grid Crop</button>
        <div class="lc-vdiv"></div>
        <button class="lc-btn" id="lc-btn-zoom-out" title="ซูมออก">−</button>
        <span id="lc-zoom-val">100%</span>
        <button class="lc-btn" id="lc-btn-zoom-in"  title="ซูมเข้า">+</button>
        <button class="lc-btn" id="lc-btn-zoom-fit" title="Fit ภาพเข้าหน้าจอ">⊡ Fit</button>
      </div>
    </div>

    <!-- Row 3 · Social Presets (wraps) -->
    <div class="lc-section lc-section-social">
      <span class="lc-section-lbl">Social</span>
      <div class="lc-content">
        <button class="lc-chip" data-ratio="4:5">📷 IG Feed</button>
        <button class="lc-chip" data-ratio="1:1">◻ Square</button>
        <button class="lc-chip" data-ratio="9:16">📱 Story/Reel</button>
        <button class="lc-chip" data-ratio="16:9">▶ YouTube</button>
        <button class="lc-chip" data-ratio="205:78">📘 FB Cover</button>
        <button class="lc-chip" data-ratio="4:1">💼 LinkedIn</button>
        <button class="lc-chip" data-ratio="2:3">📌 Pinterest</button>
        <button class="lc-chip" data-ratio="1200:630">🔗 OG Image</button>
      </div>
    </div>

  </div>
</div>
"""


def _save_dir_row(default_subdir: str, label: str = "📁 โฟลเดอร์บันทึก"):
    """Render a save-dir row. Returns (save_dir_textbox, save_status_textbox)."""
    with gr.Row():
        save_dir = gr.Textbox(
            value=_output_subdir(default_subdir),
            label=label, scale=5, lines=1,
        )
        open_btn = gr.Button("📂 เปิดโฟลเดอร์", variant="secondary", scale=1, min_width=130)
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
            "Lanczos (คุณภาพสูงสุด)": Image.LANCZOS,
            "Bicubic": Image.BICUBIC,
            "Bilinear": Image.BILINEAR,
            "Nearest (เร็วสุด)": Image.NEAREST,
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
        # ── Parse extended coords: "x,y,x2,y2|r:90|fh:1|fv:0" ──
        raw = (coords_str or "").strip()
        segments = raw.split("|")
        coord_part = segments[0]
        rotation = 0
        flip_h = False
        flip_v = False
        for seg in segments[1:]:
            if ":" in seg:
                k, v = seg.split(":", 1)
                if k == "r":  rotation = int(v)
                elif k == "fh": flip_h = bool(int(v))
                elif k == "fv": flip_v = bool(int(v))

        # ── Apply transforms to get display image ──────────────
        disp = image.copy()
        if rotation != 0:
            disp = disp.rotate(-rotation, expand=True)   # PIL is CCW, so negate for CW
        if flip_h:
            disp = disp.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v:
            disp = disp.transpose(Image.FLIP_TOP_BOTTOM)

        dw, dh = disp.size

        # ── Crop coords in display space ────────────────────────
        if coord_part and coord_part.count(",") == 3:
            parts = [int(float(v)) for v in coord_part.split(",")]
            l, t, r, b = parts
        else:
            l, t, r, b = 0, 0, dw, dh

        l = max(0, min(l, dw - 1))
        t = max(0, min(t, dh - 1))
        r = max(l + 1, min(r, dw))
        b = max(t + 1, min(b, dh))

        result = disp.crop((l, t, r, b))
        saved = _save_image(result, output_dir, "cropped", fmt=fmt)
        transform_note = ""
        if rotation or flip_h or flip_v:
            parts_note = []
            if rotation: parts_note.append(f"↻{rotation}°")
            if flip_h:   parts_note.append("↔Flip H")
            if flip_v:   parts_note.append("↕Flip V")
            transform_note = "  ·  " + " ".join(parts_note)
        return result, f"✅ Cropped {r-l}×{b-t} px{transform_note}\n💾 {saved}"
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
# LIGHTBOX — intercept Gradio's requestFullscreen → show popup
# ============================================================

# ── Lightbox via launch(js=...) ───────────────────────────────────────
# Gradio 6 injects config.js as a real <script> tag in document.head.
# (Index-CV3JHPjL.js line 555: script.textContent = get(config).js)
# Strategy: event delegation on document (capture phase) to intercept
# clicks on Gradio's fullscreen button (title="Fullscreen") BEFORE
# Gradio's own handlers. Stop propagation prevents native requestFullscreen.
# Walk up DOM from button to find img src or video src, then show lightbox.
# ─────────────────────────────────────────────────────────────────────

# LIGHTBOX_JS is injected via launch(js=...) which creates a <script> tag in document.head.
# (Confirmed in Index-CV3JHPjL.js line 555: script.textContent = get(config).js; document.head.appendChild(script))
LIGHTBOX_JS = """
(function() {
  if (window._lcFSPatched) return;
  window._lcFSPatched = true;

  function mk(tag, cls) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    return el;
  }

  function ensureLB() {
    if (document.getElementById('lc-lb')) return;
    var lb  = mk('div'); lb.id = 'lc-lb';
    var bg  = mk('div', 'lc-lb-bg');
    var box = mk('div', 'lc-lb-box');
    var cls = mk('button', 'lc-lb-close'); cls.title = 'Close (Esc)'; cls.textContent = '\\u00d7';
    var img = mk('img',  'lc-lb-img');
    var vid = mk('video','lc-lb-vid'); vid.controls = true; vid.setAttribute('playsinline','');
    box.appendChild(cls); box.appendChild(img); box.appendChild(vid);
    lb.appendChild(bg); lb.appendChild(box);
    document.body.appendChild(lb);
    bg.addEventListener('click',  closeLB);
    cls.addEventListener('click', closeLB);
    document.addEventListener('keydown', function(e){ if(e.key==='Escape') closeLB(); });
  }

  function closeLB() {
    var lb = document.getElementById('lc-lb');
    if (!lb) return;
    lb.classList.remove('lc-open');
    var v = lb.querySelector('video');
    if (v) { v.pause(); v.removeAttribute('src'); v.load(); }
  }

  function openLB(src, isVid) {
    ensureLB();
    var lb  = document.getElementById('lc-lb');
    var img = lb.querySelector('img');
    var vid = lb.querySelector('video');
    if (isVid) {
      img.style.display='none'; vid.style.display='block';
      vid.src=src; vid.load();
    } else {
      vid.style.display='none'; img.style.display='block';
      img.src=src;
    }
    lb.classList.add('lc-open');
  }

  // Intercept Gradio's fullscreen button clicks via event delegation (capture phase).
  // Gradio's fullscreen button has title="Fullscreen" and aria-label="Fullscreen".
  // We stop propagation so Gradio won't also attempt requestFullscreen natively.
  document.addEventListener('click', function(e) {
    var btn = e.target.closest
      ? e.target.closest('button[title="Fullscreen"], button[aria-label="Fullscreen"]')
      : null;
    if (!btn) return;

    // Prevent Gradio from attempting native requestFullscreen
    e.stopPropagation();

    // Walk up from the button to find the nearest image or video
    var node = btn;
    for (var depth = 0; depth < 10; depth++) {
      node = node.parentElement;
      if (!node || node === document.body) break;

      // Check for video element first
      var vid = node.querySelector('video');
      if (vid) {
        var vsrc = vid.src || ((vid.querySelector('source') || {}).src || '');
        if (vsrc) { openLB(vsrc, true); return; }
      }

      // Check for a real image (skip GIF placeholders and empty srcs)
      var imgs = node.querySelectorAll('img[src]');
      for (var i = 0; i < imgs.length; i++) {
        var s = imgs[i].src;
        if (s && s.indexOf('data:image/gif') < 0 && s !== location.href) {
          openLB(s, false); return;
        }
      }
    }
  }, true); // capture phase — fires before Gradio's own handlers

  // ── Fix Gradio tab overflow ─────────────────────────────────────────────
  // Gradio 6 uses handle_menu_overflow() which calls
  //   tab_nav_el.getBoundingClientRect().width
  // to decide which tabs fit. We override that method on the element so it
  // returns 9999, causing Gradio to mark ALL tabs as visible_tabs.
  // The "..." overflow span is a SIBLING of div[role="tablist"], not inside it.
  (function() {
    function _fixTabOverflow() {
      var tabNav = document.querySelector('[role="tablist"]');
      if (!tabNav) { setTimeout(_fixTabOverflow, 400); return; }
      if (tabNav._lcTabFixed) return;
      tabNav._lcTabFixed = true;

      // Override getBoundingClientRect so Gradio thinks all tabs fit
      tabNav.getBoundingClientRect = function() {
        var r = Element.prototype.getBoundingClientRect.call(this);
        return {
          x: r.x, y: r.y, width: 9999, height: r.height,
          top: r.top, right: r.left + 9999, bottom: r.bottom, left: r.left,
          toJSON: function() { return this; }
        };
      };

      // Hide the overflow "..." span (sibling of the tablist)
      var wrapper = tabNav.closest('.tab-wrapper');
      if (wrapper) {
        var span = wrapper.querySelector('span');
        if (span) span.style.setProperty('display', 'none', 'important');
      }

      // Trigger Gradio's overflow re-computation
      window.dispatchEvent(new Event('resize'));
    }
    setTimeout(_fixTabOverflow, 800);
  })();

})();
"""

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
    /* overflow:visible lets image/video fullscreen overlay escape the block */
    overflow: visible !important;
}

/* Re-clip only the innermost image frame — NOT .image-container.
   Gradio 6.x: .image-container holds both the <img> frame AND the
   floating toolbar (FullscreenButton/Download/Share).  Setting
   overflow:hidden on .image-container clips that toolbar, blocking clicks. */
.image-frame {
    overflow: hidden !important;
    border-radius: var(--r-sm) !important;
}
/* Keep .image-container overflow visible so toolbar buttons are not clipped */
.image-container {
    overflow: visible !important;
}

/* ── Tab nav ── */
/* The tab-wrapper has a fixed height that clips wrapped rows — override it.
   Gradio CSS uses .tab-wrapper.svelte-11gaq1 (specificity 0,2,0).
   We beat it with div.tab-wrapper + !important (same specificity, later source). */
.tab-wrapper.svelte-11gaq1 {
    height: auto !important;
    padding-bottom: 0 !important;
    flex-wrap: wrap !important;
}
/* The tablist itself must allow wrapping and be visible */
div[role="tablist"] {
    background: var(--bg-base) !important;
    border-bottom: 1px solid var(--bd) !important;
    padding: 4px 12px 0 !important;
    gap: 0 !important;
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    overflow: visible !important;
    height: auto !important;
    width: 100% !important;
}
/* Hide Gradio's "..." overflow span (it is a SIBLING of div[role="tablist"],
   not inside it — so target it as .tab-wrapper > span) */
.tab-wrapper.svelte-11gaq1 > span,
.tab-wrapper > span:last-child {
    display: none !important;
}
/* Hide the internal ::after border line — our div[role="tablist"] border-bottom covers it */
.tab-container.svelte-11gaq1::after {
    display: none !important;
}

button[role="tab"] {
    display: flex !important;
    background: transparent !important;
    color: var(--tx-muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 14px 8px !important;
    font-weight: 600 !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.06em !important;
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

/* ── Download info box ── */
.lc-info-box {
    background: rgba(220,38,38,0.06) !important;
    border: 1px solid rgba(220,38,38,0.18) !important;
    border-radius: var(--r-md) !important;
    padding: 12px 16px !important;
    font-size: 0.83rem !important;
    color: var(--tx-muted) !important;
    line-height: 1.7 !important;
    margin-bottom: 8px !important;
}

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

/* ════════════════════════════════════════════════════════
   FIX: Gradio 6.x native fullscreen (requestFullscreen API)
   Source confirmed: Index-D6KKbqPS.js calls
     get(image_container).requestFullscreen?.()
   The image_container div holds BOTH the image AND the toolbar.
   Any overflow:hidden or pointer-events issue on image_container
   will block the fullscreen button click or clip the toolbar.
════════════════════════════════════════════════════════ */

/* Gradio 6.x :fullscreen state — keep black bg, centred */
.image-container:fullscreen {
    background: #000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.image-container:fullscreen img,
.image-container:fullscreen video {
    max-width:  100vw !important;
    max-height: 100vh !important;
    object-fit: contain !important;
    border-radius: 0 !important;
}

/* Toolbar (ActionButtonWrapper) inside .image-container — must be visible */
.image-container button,
.image-container svg,
.image-container [class*="button"] {
    pointer-events: all !important;
}

/* Video fullscreen — same pattern */
.video-container:fullscreen {
    background: #000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.video-container:fullscreen video {
    max-width:  100vw !important;
    max-height: 100vh !important;
    object-fit: contain !important;
}

/* ════════════════════════════════════════════
   LIGHTBOX POPUP (replaces native fullscreen)
════════════════════════════════════════════ */
#lc-lb {
    display: none;
    position: fixed; inset: 0; z-index: 999999;
    align-items: center; justify-content: center;
}
#lc-lb.lc-open { display: flex; }

.lc-lb-bg {
    position: absolute; inset: 0;
    background: rgba(0,0,0,0.88);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    cursor: zoom-out;
}

.lc-lb-box {
    position: relative; z-index: 1;
    display: flex; align-items: center; justify-content: center;
    max-width: 94vw; max-height: 94vh;
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 40px 100px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.06);
    animation: lc-lb-in 0.2s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes lc-lb-in {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
}

.lc-lb-img,
.lc-lb-vid {
    display: block;
    max-width: 90vw;
    max-height: 90vh;
    object-fit: contain;
    border-radius: 0;
}

.lc-lb-close {
    position: absolute; top: 12px; right: 14px; z-index: 2;
    width: 36px; height: 36px;
    background: rgba(10,10,10,0.75);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50%;
    color: #fff; font-size: 15px; font-weight: 700;
    cursor: pointer; line-height: 1;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s, transform 0.15s;
}
.lc-lb-close:hover {
    background: rgba(220,38,38,0.85);
    transform: scale(1.1);
}

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

  #lc-hint      { display: none !important; }
}
"""


def _get_gpu_badge_html():
    """Return an HTML badge showing GPU/CPU status for the header."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            # Green badge — GPU active
            return (
                f'<span title="GPU acceleration active" '
                f'style="display:inline-flex;align-items:center;gap:5px;'
                f'padding:4px 10px;border-radius:9999px;'
                f'background:#052e16;border:1px solid #16a34a;'
                f'font-size:0.6rem;font-weight:700;color:#4ade80;'
                f'letter-spacing:0.04em;white-space:nowrap;flex-shrink:0;">'
                f'🟢 GPU · {name} · {vram}GB</span>'
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return (
                '<span title="Apple Silicon GPU active" '
                'style="display:inline-flex;align-items:center;gap:5px;'
                'padding:4px 10px;border-radius:9999px;'
                'background:#052e16;border:1px solid #16a34a;'
                'font-size:0.6rem;font-weight:700;color:#4ade80;'
                'letter-spacing:0.04em;white-space:nowrap;flex-shrink:0;">'
                '🟢 MPS · Apple Silicon</span>'
            )
        else:
            ver = torch.__version__
            reason = "CPU-only torch" if "+cpu" in ver else "CUDA unavailable"
            return (
                f'<span title="GPU not available — {reason}" '
                f'style="display:inline-flex;align-items:center;gap:5px;'
                f'padding:4px 10px;border-radius:9999px;'
                f'background:#1c0000;border:1px solid #b91c1c;'
                f'font-size:0.6rem;font-weight:700;color:#f87171;'
                f'letter-spacing:0.04em;white-space:nowrap;flex-shrink:0;">'
                f'🔴 CPU only · {reason}</span>'
            )
    except Exception:
        return (
            '<span style="display:inline-flex;align-items:center;gap:5px;'
            'padding:4px 10px;border-radius:9999px;'
            'background:#1c0000;border:1px solid #b91c1c;'
            'font-size:0.6rem;font-weight:700;color:#f87171;">'
            '🔴 torch unavailable</span>'
        )


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
        # right: GPU status badge
        '<div style="flex-shrink:0;">'
        + _get_gpu_badge_html() +
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
⚠️ Video upscaling is GPU-intensive — processing may take several minutes per minute of video. GPU strongly recommended.<br>
📌 <b>General (Fast)</b> and <b>Anime / Cartoon</b> are always 4x — Scale Factor selector applies to <b>General (Best Quality)</b> only.
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
    var wrapEl = canvas.parentElement;
    var maxW   = (wrapEl ? wrapEl.clientWidth : 600) - 14;
    if (!maxW || maxW < 10) maxW = 600;
    // maxH = longest side of image scaled to fit available width
    // (keeps wrap stable during rotation — neither portrait nor landscape overflows)
    var _longSide = Math.max(origW, origH);
    var maxH      = Math.round(_longSide * Math.min(1, maxW / _longSide));
    if (maxH < 160) maxH = 160;
    // min-height only — wrap can still grow when zooming
    if (wrapEl) wrapEl.style.minHeight = (maxH + 14) + 'px';
    var scale = Math.min(1, maxW / origW, maxH / origH);

    // ── Zoom ──────────────────────────────────────────────
    var baseScale  = scale;
    var zoomSteps  = [0.25,0.33,0.5,0.67,0.75,1.0,1.25,1.5,2.0,2.5,3.0];
    var _savedZoom = parseFloat(localStorage.getItem('lc_crop_zoom'));
    var zoomFactor = (_savedZoom && zoomSteps.indexOf(_savedZoom) >= 0) ? _savedZoom : 1.0;
    canvas.width  = Math.round(origW * baseScale * zoomFactor);
    canvas.height = Math.round(origH * baseScale * zoomFactor);

    // ── Transform state ───────────────────────────────────
    var rotation = 0;      // 0 | 90 | 180 | 270  (degrees CW)
    var flipH    = false;
    var flipV    = false;

    // canvas dims depend on rotation (90/270 swap w/h)
    function computeCanvasDims(){
      var s=baseScale*zoomFactor;
      return (rotation===90||rotation===270)
        ? {w:Math.round(origH*s), h:Math.round(origW*s)}
        : {w:Math.round(origW*s), h:Math.round(origH*s)};
    }
    // logical display image size (after rotation)
    function displayDims(){
      return (rotation===90||rotation===270)
        ? {w:origH, h:origW}
        : {w:origW, h:origH};
    }

    function applyZoom(newFactor){
      var oldW=canvas.width, oldH=canvas.height;
      zoomFactor=newFactor;
      var d=computeCanvasDims();
      if(hasSel && oldW>0){
        var fx=d.w/oldW, fy=d.h/oldH;
        sx=Math.round(sx*fx); sy=Math.round(sy*fy);
        ex=Math.round(ex*fx); ey=Math.round(ey*fy);
      }
      canvas.width=d.w; canvas.height=d.h;
      redraw(); updateCoords();
      var zEl=document.getElementById('lc-zoom-val');
      if(zEl) zEl.textContent=Math.round(zoomFactor*100)+'%';
      try{ localStorage.setItem('lc_crop_zoom', zoomFactor); }catch(e){}
    }

    // ── State ─────────────────────────────────────────────
    var sx=0, sy=0, ex=0, ey=0;
    var hasSel = false;
    var isDown = false;
    var dragMode = 'draw';          // 'draw' | 'move' | 'resize-<id>'
    var dragStart = null;           // snapshot of sx/sy/ex/ey at mousedown
    var lockedRatio = null;         // null = free  |  {w, h}
    var gridMode    = 0; // 0=off  1=rule-of-thirds  2=golden-ratio(φ)  3=diagonal+cross  (crop box)
    var imgGridMode = 0; // same cycle but for the full canvas image
    var HR = 4;                     // handle radius px

    // ── RAF throttle — prevents flickering during drag ─────
    var _rafId = null;
    function scheduleRedraw() {
      if (_rafId) return;
      _rafId = requestAnimationFrame(function() { _rafId = null; redraw(); });
    }
    var _coordsTimer = null;
    function scheduleCoords() {
      if (_coordsTimer) return;
      _coordsTimer = setTimeout(function() { _coordsTimer = null; updateCoords(); }, 60);
    }

    // ── Helpers ───────────────────────────────────────────
    function getPos(e) {
      var r = canvas.getBoundingClientRect();
      var sc = canvas.width / (canvas.clientWidth || canvas.width);
      var src = (e.touches && e.touches[0]) ? e.touches[0] : e;
      return { x: Math.round((src.clientX-r.left)*sc),
               y: Math.round((src.clientY-r.top)*sc) };
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
    function drawTransformed(){
      var s=baseScale*zoomFactor;
      var iw=Math.round(origW*s), ih=Math.round(origH*s);
      var cw=canvas.width, ch=canvas.height;
      ctx.save();
      // rotation
      if(rotation===0){}
      else if(rotation===90){ ctx.translate(cw,0); ctx.rotate(Math.PI/2); }
      else if(rotation===180){ ctx.translate(cw,ch); ctx.rotate(Math.PI); }
      else if(rotation===270){ ctx.translate(0,ch); ctx.rotate(-Math.PI/2); }
      // flip (applied in rotated/display space)
      if(flipH){ ctx.translate(rotation===90||rotation===270?ih:iw,0); ctx.scale(-1,1); }
      if(flipV){ ctx.translate(0,rotation===90||rotation===270?iw:ih); ctx.scale(1,-1); }
      ctx.drawImage(img,0,0,iw,ih);
      ctx.restore();
    }
    function redraw() {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      drawTransformed();
      // ── Full-image grid overlay ───────────────────────────
      if(imgGridMode > 0) {
        ctx.save();
        ctx.strokeStyle='rgba(255,255,255,0.50)'; ctx.lineWidth=1; ctx.setLineDash([3,4]);
        ctx.beginPath();
        if(imgGridMode === 1) {
          var gx1=canvas.width/3, gx2=2*canvas.width/3;
          var gy1=canvas.height/3, gy2=2*canvas.height/3;
          ctx.moveTo(gx1,0); ctx.lineTo(gx1,canvas.height);
          ctx.moveTo(gx2,0); ctx.lineTo(gx2,canvas.height);
          ctx.moveTo(0,gy1); ctx.lineTo(canvas.width,gy1);
          ctx.moveTo(0,gy2); ctx.lineTo(canvas.width,gy2);
        } else if(imgGridMode === 2) {
          var phi=0.618;
          var ipx1=canvas.width*(1-phi), ipx2=canvas.width*phi;
          var ipy1=canvas.height*(1-phi), ipy2=canvas.height*phi;
          ctx.moveTo(ipx1,0); ctx.lineTo(ipx1,canvas.height);
          ctx.moveTo(ipx2,0); ctx.lineTo(ipx2,canvas.height);
          ctx.moveTo(0,ipy1); ctx.lineTo(canvas.width,ipy1);
          ctx.moveTo(0,ipy2); ctx.lineTo(canvas.width,ipy2);
        } else if(imgGridMode === 3) {
          ctx.moveTo(0,0); ctx.lineTo(canvas.width,canvas.height);
          ctx.moveTo(canvas.width,0); ctx.lineTo(0,canvas.height);
          ctx.moveTo(canvas.width/2,0); ctx.lineTo(canvas.width/2,canvas.height);
          ctx.moveTo(0,canvas.height/2); ctx.lineTo(canvas.width,canvas.height/2);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }
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
      // grid overlay — cycles through 3 modes
      if(gridMode > 0) {
        ctx.strokeStyle='rgba(255,255,255,0.55)'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        ctx.beginPath();
        if(gridMode === 1) {
          // Rule of Thirds — divide at 1/3 and 2/3
          var gx1=r.x+r.w/3, gx2=r.x+2*r.w/3, gy1=r.y+r.h/3, gy2=r.y+2*r.h/3;
          ctx.moveTo(gx1,r.y); ctx.lineTo(gx1,r.y+r.h);
          ctx.moveTo(gx2,r.y); ctx.lineTo(gx2,r.y+r.h);
          ctx.moveTo(r.x,gy1); ctx.lineTo(r.x+r.w,gy1);
          ctx.moveTo(r.x,gy2); ctx.lineTo(r.x+r.w,gy2);
        } else if(gridMode === 2) {
          // Golden Ratio — divide at φ≈0.618 and 1-φ≈0.382
          var phi=0.618;
          var px1=r.x+r.w*(1-phi), px2=r.x+r.w*phi, py1=r.y+r.h*(1-phi), py2=r.y+r.h*phi;
          ctx.moveTo(px1,r.y); ctx.lineTo(px1,r.y+r.h);
          ctx.moveTo(px2,r.y); ctx.lineTo(px2,r.y+r.h);
          ctx.moveTo(r.x,py1); ctx.lineTo(r.x+r.w,py1);
          ctx.moveTo(r.x,py2); ctx.lineTo(r.x+r.w,py2);
        } else if(gridMode === 3) {
          // Diagonal + center cross — good for dynamic/symmetry composition
          ctx.moveTo(r.x,r.y); ctx.lineTo(r.x+r.w,r.y+r.h);
          ctx.moveTo(r.x+r.w,r.y); ctx.lineTo(r.x,r.y+r.h);
          ctx.moveTo(r.x+r.w/2,r.y); ctx.lineTo(r.x+r.w/2,r.y+r.h);
          ctx.moveTo(r.x,r.y+r.h/2); ctx.lineTo(r.x+r.w,r.y+r.h/2);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // handles
      ctx.setLineDash([]);
      var hs=handles();
      for(var i=0;i<hs.length;i++){
        ctx.beginPath(); ctx.arc(hs[i].x,hs[i].y,HR,0,2*Math.PI);
        ctx.fillStyle='#111'; ctx.fill();
        ctx.strokeStyle='rgba(255,255,255,0.55)'; ctx.lineWidth=1; ctx.stroke();
      }
      ctx.restore();
    }

    // ── Coords Update ─────────────────────────────────────
    function updateCoords() {
      var sc=1/(baseScale*zoomFactor);  // canvas px → display image px
      var d=displayDims();
      var x0,y0,x1,y1;
      if(hasSel && (Math.abs(ex-sx)>1 || Math.abs(ey-sy)>1)){
        x0=Math.max(0,Math.round(Math.min(sx,ex)*sc));
        y0=Math.max(0,Math.round(Math.min(sy,ey)*sc));
        x1=Math.min(d.w,Math.round(Math.max(sx,ex)*sc));
        y1=Math.min(d.h,Math.round(Math.max(sy,ey)*sc));
      } else {
        // no selection → full image (still encode transforms)
        x0=0; y0=0; x1=d.w; y1=d.h;
      }
      var coords=x0+','+y0+','+x1+','+y1
        +'|r:'+rotation+'|fh:'+(flipH?1:0)+'|fv:'+(flipV?1:0);
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
      var xfStr=(rotation?'↻'+rotation+'° ':'')+(flipH?'↔H ':'')+(flipV?'↕V ':'');
      var info=document.getElementById('lc-crop-info');
      if(info) info.textContent=(x1-x0)+'×'+(y1-y0)+' px  ·  ('+x0+', '+y0+')  ·  '+rStr+(xfStr?' · '+xfStr.trim():'')+'  ·  '+d.w+'×'+d.h+' px';
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
      scheduleRedraw();
      if(hasSel || dragMode!=='draw') scheduleCoords();
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

    // ── Touch events (mobile) ─────────────────────────────
    canvas.addEventListener('touchstart', function(e){
      e.preventDefault(); canvas.onmousedown(e.touches[0]);
    }, {passive:false});
    canvas.addEventListener('touchmove', function(e){
      e.preventDefault(); canvas.onmousemove(e);
    }, {passive:false});
    canvas.addEventListener('touchend', function(e){
      e.preventDefault(); canvas.onmouseup(e.changedTouches ? {clientX:e.changedTouches[0].clientX, clientY:e.changedTouches[0].clientY} : e);
    }, {passive:false});

    // ── Toolbar Buttons ───────────────────────────────────
    function setRatio(rStr) {
      if(rStr==='free'){ lockedRatio=null; return; }
      if(rStr==='orig'){ var dd=displayDims(); lockedRatio={w:dd.w,h:dd.h}; }
      else { var p=rStr.split(':'); lockedRatio={w:parseFloat(p[0]),h:parseFloat(p[1])}; }
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

    // Ratio buttons (support both lc-btn-pill and legacy lc-ratio-btn with data-ratio)
    var RATIO_SEL = '.lc-btn-pill[data-ratio], .lc-ratio-btn[data-ratio]';
    function clearRatioActive(){
      document.querySelectorAll(RATIO_SEL).forEach(function(b){b.classList.remove('lc-active');});
      document.querySelectorAll('.lc-chip').forEach(function(b){b.classList.remove('lc-active');});
    }
    document.querySelectorAll(RATIO_SEL).forEach(function(btn){
      btn.addEventListener('click',function(){
        clearRatioActive();
        btn.classList.add('lc-active');
        setRatio(btn.getAttribute('data-ratio'));
      });
    });

    // Social presets
    document.querySelectorAll('.lc-chip').forEach(function(btn){
      btn.addEventListener('click',function(){
        clearRatioActive();
        btn.classList.add('lc-active');
        var p=btn.getAttribute('data-ratio').split(':');
        centerAndApplyRatio(parseFloat(p[0]),parseFloat(p[1]));
      });
    });

    // Crop-box grid cycle: Off → 3×3 → φ → Diagonal → Off
    var GRID_LABELS = ['⊞ Grid Crop', '⊞ 3×3', '⊞ φ', '⊞ ✕'];
    var GRID_TITLES = ['Grid ในกรอบ Crop', 'Rule of Thirds (3×3)', 'Golden Ratio (φ)', 'Diagonal + Center'];
    var gBtn=document.getElementById('lc-btn-grid');
    if(gBtn) gBtn.addEventListener('click',function(){
      gridMode = (gridMode + 1) % 4;
      gBtn.textContent = GRID_LABELS[gridMode];
      gBtn.title = GRID_TITLES[gridMode];
      gBtn.classList.toggle('lc-active', gridMode > 0);
      redraw();
    });

    // Full-image grid cycle: Off → 3×3 → φ → Diagonal → Off
    var IMG_GRID_LABELS = ['⊟ Grid ภาพ', '⊟ 3×3', '⊟ φ', '⊟ ✕'];
    var IMG_GRID_TITLES = ['Grid ทั้งภาพ', 'Rule of Thirds (ทั้งภาพ)', 'Golden Ratio (ทั้งภาพ)', 'Diagonal + Center (ทั้งภาพ)'];
    var igBtn=document.getElementById('lc-btn-img-grid');
    if(igBtn) igBtn.addEventListener('click',function(){
      imgGridMode = (imgGridMode + 1) % 4;
      igBtn.textContent = IMG_GRID_LABELS[imgGridMode];
      igBtn.title = IMG_GRID_TITLES[imgGridMode];
      igBtn.classList.toggle('lc-active', imgGridMode > 0);
      redraw();
    });

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
      if(info){ var _d=displayDims(); info.textContent='ลากเพื่อเลือกพื้นที่ · '+_d.w+'×'+_d.h+' px'; }
    });

    // ── Transform (Rotate / Flip) ─────────────────────────
    function applyTransform(type){
      // reset selection — canvas geometry may change on rotation
      sx=0;sy=0;ex=0;ey=0;hasSel=false;
      if(type==='rot-l')        rotation=(rotation+270)%360;
      else if(type==='rot-r')   rotation=(rotation+90)%360;
      else if(type==='rot-180') rotation=(rotation+180)%360;
      else if(type==='flip-h')  flipH=!flipH;
      else if(type==='flip-v')  flipV=!flipV;
      else if(type==='xform-reset'){ rotation=0; flipH=false; flipV=false; }
      // Refit: recalculate baseScale for new orientation (keeps image fully visible)
      // zoomFactor intentionally preserved — user's zoom level survives rotation
      var dd2=displayDims();
      baseScale=Math.min(1, maxW/dd2.w, maxH/dd2.h);
      var zEl=document.getElementById('lc-zoom-val');
      if(zEl) zEl.textContent=Math.round(zoomFactor*100)+'%';
      // resize canvas for new rotation
      var d=computeCanvasDims();
      canvas.width=d.w; canvas.height=d.h;
      // update flip button active state
      var fhB=document.getElementById('lc-btn-flip-h');
      var fvB=document.getElementById('lc-btn-flip-v');
      if(fhB) fhB.classList.toggle('lc-active',flipH);
      if(fvB) fvB.classList.toggle('lc-active',flipV);
      // Use RAF so redraw fires AFTER layout reflow from canvas resize settles
      // (prevents blank-canvas flash that ResizeObserver/fitToolbar can cause)
      if(_rafId) cancelAnimationFrame(_rafId);
      _rafId = requestAnimationFrame(function(){ _rafId=null; redraw(); updateCoords(); });
      var dd=displayDims();
      var info=document.getElementById('lc-crop-info');
      if(info) info.textContent='ลากเพื่อเลือกพื้นที่ · '+dd.w+'×'+dd.h+' px'
        +(rotation?' · ↻'+rotation+'°':'')+(flipH?' · ↔H':'')+(flipV?' · ↕V':'');
    }
    ['rot-l','rot-r','rot-180','flip-h','flip-v','xform-reset'].forEach(function(id){
      var b=document.getElementById('lc-btn-'+id);
      if(b) b.addEventListener('click',function(){ applyTransform(id); });
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
      if(info){ var _d=displayDims(); info.textContent='ลากเพื่อเลือกพื้นที่ · '+_d.w+'×'+_d.h+' px'; }
    }
    if(img.complete){ initDraw(); } else { img.onload=initDraw; }

    // ── Force toolbar button sizes (JS wins over any Gradio CSS) ──────────
    function _fitToolbar() {
      var toolbar = document.getElementById('lc-toolbar');
      if (!toolbar) return;
      var cw = toolbar.clientWidth;
      if (!cw || cw < 10) return;
      // Scale font & padding relative to actual toolbar pixel width
      // At cw=800: font=13px pad=6px 11px  |  cw=500: font=11px pad=5px 9px  |  cw=320: font=10px pad=4px 7px
      var fs  = Math.max(10, Math.min(13, cw * 0.019));   // 10–13 px
      var px  = Math.max(6,  Math.min(11, cw * 0.015));   // 6–11 px horizontal pad
      var py  = Math.max(4,  Math.min(6,  cw * 0.009));   // 4–6 px vertical pad
      var fsS = fs.toFixed(1)+'px';
      var pad = py.toFixed(1)+'px '+px.toFixed(1)+'px';
      toolbar.querySelectorAll('button').forEach(function(b) {
        b.style.setProperty('font-size',          fsS,  'important');
        b.style.setProperty('padding',            pad,  'important');
        b.style.setProperty('font-weight',        '600','important');
        b.style.setProperty('line-height',        '1.2','important');
        b.style.setProperty('white-space',        'nowrap','important');
        b.style.setProperty('box-sizing',         'border-box','important');
        b.style.setProperty('-webkit-appearance', 'none','important');
        b.style.setProperty('appearance',         'none','important');
        b.style.setProperty('margin',             '0',  'important');
      });
      // Zoom label
      var zv = document.getElementById('lc-zoom-val');
      if (zv) {
        zv.style.setProperty('font-size', fsS, 'important');
        zv.style.setProperty('min-width', Math.round(cw*0.04)+'px', 'important');
      }
      // Section labels
      toolbar.querySelectorAll('.lc-section-lbl').forEach(function(el){
        var lblW = Math.max(36, Math.min(54, Math.round(cw*0.075)));
        el.style.setProperty('width',     lblW+'px','important');
        el.style.setProperty('min-width', lblW+'px','important');
        el.style.setProperty('font-size', Math.max(7, Math.min(9, cw*0.011)).toFixed(1)+'px','important');
      });
    }
    _fitToolbar();
    // Sync zoom label with restored zoom value
    var _zvInit = document.getElementById('lc-zoom-val');
    if (_zvInit) _zvInit.textContent = Math.round(zoomFactor*100)+'%';
    // Re-fit on resize (container-aware, not just window)
    if (window.ResizeObserver) {
      new ResizeObserver(function(){ _fitToolbar(); })
        .observe(document.getElementById('lc-toolbar') || document.body);
    } else {
      window.addEventListener('resize', _fitToolbar);
    }

  }, 150);
}
"""


def _build_download_tab(cfg: dict):
    """YouTube / Facebook Reels / Instagram / TikTok video downloader tab."""

    # ── Shared state between generator and pause/stop handlers ───────────────
    _current: dict = {"state": None}

    class _StopDownload(BaseException):
        """Raised inside yt-dlp progress hook to abort download."""

    # ── Helper: get imageio-ffmpeg binary ────────────────────────────────────
    def _ffmpeg_bin():
        try:
            import imageio_ffmpeg as _iio
            return _iio.get_ffmpeg_exe()
        except Exception:
            return None

    # ── Helper: base yt-dlp options ──────────────────────────────────────────
    def _base_ydl_opts():
        import shutil as _sh
        opts: dict = {"quiet": True, "no_warnings": False, "color": False}
        ffmpeg = _ffmpeg_bin()
        if ffmpeg:
            opts["ffmpeg_location"] = ffmpeg
        for bin_name, rt_key in [("node", "nodejs"), ("nodejs", "nodejs"), ("deno", "deno")]:
            p = _sh.which(bin_name)
            if p:
                opts["js_runtimes"] = {rt_key: {"path": p}}
                break
        return opts

    # ── Step 1: Fetch quality list ────────────────────────────────────────────
    def _fetch(url):
        url = (url or "").strip()
        if not url or not url.startswith(("http://", "https://")):
            return (
                gr.update(value="❌ URL ไม่ถูกต้อง — ต้องขึ้นต้นด้วย http:// หรือ https://", visible=True),
                gr.update(choices=[], visible=False),
                {},
                gr.update(visible=False),
            )
        try:
            import yt_dlp
        except ImportError:
            return (
                gr.update(value="❌ ไม่พบ yt-dlp — กรุณากด Fix แล้ว Start ใหม่", visible=True),
                gr.update(choices=[], visible=False),
                {},
                gr.update(visible=False),
            )

        ydl_opts = {**_base_ydl_opts(), "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            return (
                gr.update(value=f"❌ ดึงข้อมูลไม่สำเร็จ:\n{str(e)[-400:]}", visible=True),
                gr.update(choices=[], visible=False),
                {},
                gr.update(visible=False),
            )

        title    = info.get("title", "Unknown")
        duration = info.get("duration") or 0
        dur_str  = f"{int(duration)//60}:{int(duration)%60:02d}" if duration else "?"
        uploader = info.get("uploader") or info.get("channel") or ""

        formats = info.get("formats") or []
        heights = sorted(
            {f["height"] for f in formats
             if f.get("height") and f.get("vcodec", "none") != "none" and f["height"] > 0},
            reverse=True,
        )

        HEIGHT_LABEL = {
            2160: "4K (2160p)", 1440: "2K (1440p)", 1080: "Full HD (1080p)",
            720:  "HD (720p)",  480:  "SD (480p)",  360:  "360p",
            240:  "240p",       144:  "144p",
        }

        quality_map:  dict = {}
        quality_list: list = []

        lbl = "🏆 ดีที่สุด (อัตโนมัติ)"
        quality_map[lbl] = (
            "bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        )
        quality_list.append(lbl)

        for h in heights:
            lbl = f"📹 {HEIGHT_LABEL.get(h, str(h)+'p')}"
            quality_map[lbl] = (
                f"bestvideo[height<={h}][vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]"
                f"/bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]"
                f"/bestvideo[height<={h}]+bestaudio/best[height<={h}]/best"
            )
            quality_list.append(lbl)

        lbl = "🎵 Audio เท่านั้น (MP3)"
        quality_map[lbl] = "bestaudio/best"
        quality_list.append(lbl)

        h_list   = ", ".join(str(h) + "p" for h in heights) if heights else "ไม่ทราบ"
        info_txt = f"📺 {title}"
        if uploader:
            info_txt += f"  ·  {uploader}"
        info_txt += f"\n⏱ {dur_str}  ·  {len(heights)} ความละเอียด: {h_list}"

        return (
            gr.update(value=info_txt, visible=True),
            gr.update(choices=quality_list, value=quality_list[0], visible=True),
            quality_map,
            gr.update(visible=True),
        )

    # ── Step 2: Download with real-time progress ──────────────────────────────
    def _download(url, selected, quality_map, save_dir):
        import threading, time, re as _re

        _noop       = gr.update()
        _btn_dl_on  = gr.update(interactive=False, visible=True)
        _btn_dl_off = gr.update(interactive=True,  visible=True)
        _vis_on     = gr.update(visible=True)
        _vis_off    = gr.update(visible=False)

        def _err(msg):
            return msg, _noop, _noop, _noop, _btn_dl_off, _vis_off, _vis_off

        url = (url or "").strip()
        if not url:
            yield _err("❌ กรุณาใส่ URL แล้วกด ดึงข้อมูล ก่อน"); return
        if not selected or not quality_map:
            yield _err("❌ กรุณากด ดึงข้อมูล แล้วเลือกความละเอียดก่อน"); return

        try:
            import yt_dlp
        except ImportError:
            yield _err("❌ ไม่พบ yt-dlp — กรุณากด Fix แล้ว Start ใหม่"); return

        fmt_str  = quality_map.get(selected, "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best")
        is_audio = selected.startswith("🎵")
        out_dir  = (save_dir or "").strip() or str(_OUTPUT_ROOT / "download")
        os.makedirs(out_dir, exist_ok=True)
        out_tmpl = os.path.join(
            out_dir,
            "%(title).80s_audio.%(ext)s" if is_audio else "%(title).80s_%(height)sp.%(ext)s"
        )

        state = {
            "line": "⏳ กำลังเตรียม...", "done": False,
            "error": None, "info": None,
            "stop": False, "paused": False, "stopped_by_user": False,
        }
        _current["state"] = state

        _ansi = _re.compile(r"\x1b\[[0-9;]*m")
        def _clean(s): return _ansi.sub("", s or "").strip()

        def _hook(d):
            while state["paused"] and not state["stop"]:
                time.sleep(0.1)
            if state["stop"]:
                raise _StopDownload()

            status = d.get("status", "")
            if status == "downloading":
                pct   = _clean(d.get("_percent_str")   or "?%")
                speed = _clean(d.get("_speed_str")     or "?")
                eta   = _clean(d.get("_eta_str")        or "?")
                doneb = _clean(d.get("_downloaded_bytes_str") or "?")
                totb  = _clean(d.get("_total_bytes_str") or
                               d.get("_total_bytes_estimate_str") or "?")
                try:
                    filled = int(float(pct.replace("%", "")) / 5)
                    bar = "█" * filled + "░" * (20 - filled)
                except Exception:
                    bar = "░" * 20
                pause_note = "  ⏸ พักอยู่" if state["paused"] else ""
                state["line"] = (
                    f"⬇️  [{bar}] {pct}{pause_note}\n"
                    f"📦 {doneb} / {totb}   🚀 {speed}   ⏱ ETA {eta}"
                )
            elif status == "finished":
                state["line"] = (
                    f"✅ ดาวน์โหลดไฟล์เสร็จ — กำลัง merge/convert...\n"
                    f"📄 {os.path.basename(d.get('filename', ''))}"
                )

        ydl_opts = {
            **_base_ydl_opts(),
            "format": fmt_str, "outtmpl": out_tmpl,
            "noplaylist": True, "progress_hooks": [_hook],
            "format_sort": ["vcodec:h264", "acodec:aac", "ext:mp4:m4a"],
            "prefer_free_formats": False,
        }
        if not is_audio:
            ydl_opts["merge_output_format"] = "mp4"
        else:
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }]

        def _run():
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    state["info"] = ydl.extract_info(url, download=True)
            except _StopDownload:
                state["stopped_by_user"] = True
            except Exception as e:
                state["error"] = str(e)
            finally:
                state["done"] = True

        threading.Thread(target=_run, daemon=True).start()

        try:
            while not state["done"]:
                yield state["line"], _noop, _noop, _noop, _btn_dl_on, _vis_on, _vis_on
                time.sleep(0.4)

            if state["stopped_by_user"]:
                yield "⏹ หยุดดาวน์โหลดแล้ว", _noop, _noop, _noop, _btn_dl_off, _vis_off, _vis_off
                return

            if state["error"]:
                yield (f"❌ ดาวน์โหลดไม่สำเร็จ:\n{state['error'][-500:]}",
                       _noop, _noop, _noop, _btn_dl_off, _vis_off, _vis_off)
                return

            all_files = [
                os.path.join(out_dir, f)
                for f in os.listdir(out_dir)
                if os.path.isfile(os.path.join(out_dir, f))
            ]
            if not all_files:
                yield "⚠️ ดาวน์โหลดสำเร็จแต่หาไฟล์ไม่พบ", _noop, _noop, _noop, _btn_dl_off, _vis_off, _vis_off
                return

            filepath = max(all_files, key=os.path.getmtime)
            size_mb  = os.path.getsize(filepath) / 1_048_576
            fname    = os.path.basename(filepath)
            ext      = os.path.splitext(fname)[1].upper().lstrip(".")
            title    = (state["info"] or {}).get("title", "video")
            msg      = f"✅ {title}\n📁 {fname}  ({size_mb:.1f} MB)  [{ext}]\n📂 {out_dir}"

            if is_audio:
                yield (msg,
                       gr.update(value=None,     visible=False),
                       gr.update(value=filepath, visible=True),
                       filepath, _btn_dl_off, _vis_off, _vis_off)
            else:
                yield (msg,
                       gr.update(value=filepath, visible=True),
                       gr.update(value=None,     visible=False),
                       filepath, _btn_dl_off, _vis_off, _vis_off)

        except Exception as e:
            yield f"❌ {e}", _noop, _noop, _noop, _btn_dl_off, _vis_off, _vis_off
        finally:
            _current["state"] = None

    # ── Pause toggle ──────────────────────────────────────────────────────────
    def _pause_toggle():
        s = _current.get("state")
        if not s:
            return gr.update()
        s["paused"] = not s["paused"]
        return gr.update(value="▶ ต่อ" if s["paused"] else "⏸ พัก")

    # ── Stop ──────────────────────────────────────────────────────────────────
    def _stop_download():
        s = _current.get("state")
        if s:
            s["stop"] = True

    # ── UI layout ─────────────────────────────────────────────────────────────
    gr.HTML('<div class="sec-head">ดาวน์โหลดวีดีโอ · yt-dlp</div>')
    gr.HTML(
        '<div class="lc-info-box">'
        '📋 วิธีใช้: วางลิ้งก์ → กด <b>ดึงข้อมูล</b> → เลือกความละเอียด → กด <b>ดาวน์โหลด</b><br>'
        '🌐 รองรับ: YouTube, Facebook Reels, Instagram, TikTok และอีกกว่า 1,000 เว็บไซต์'
        '</div>'
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="URL วีดีโอ",
            placeholder="https://www.youtube.com/watch?v=...  หรือ  https://www.facebook.com/reel/...",
            lines=1, scale=5,
        )
        fetch_btn = gr.Button("🔍 ดึงข้อมูล", variant="secondary", scale=1, min_width=130)

    info_out = gr.Textbox(label="ข้อมูลวีดีโอ", interactive=False, lines=3, visible=False)

    quality_dd    = gr.Dropdown(label="เลือกความละเอียด", choices=[], visible=False, interactive=True)
    quality_state = gr.State({})

    dl_out_dir, _ = _save_dir_row("download", label="📁 บันทึกวีดีโอที่")
    dl_out_dir.value = cfg["dl_out_dir"]

    with gr.Row():
        download_btn = gr.Button("⬇️ ดาวน์โหลด", variant="primary",  visible=False, scale=4)
        pause_btn    = gr.Button("⏸ พัก",         variant="secondary", visible=False, scale=1, min_width=100)
        stop_btn     = gr.Button("⏹ หยุด",        variant="stop",      visible=False, scale=1, min_width=100)

    dl_status = gr.Textbox(label="สถานะ", interactive=False, lines=3)

    with gr.Row():
        video_out = gr.Video(label="วีดีโอที่ดาวน์โหลด", interactive=False, visible=True)
        audio_out = gr.Audio(label="🎵 ฟังเพลง (Audio Only)", interactive=False, visible=False)

    file_out = gr.File(label="⬇️ บันทึกไฟล์", interactive=False)

    # ── Wire events ───────────────────────────────────────────────────────────
    fetch_btn.click(
        fn=_fetch,
        inputs=[url_input],
        outputs=[info_out, quality_dd, quality_state, download_btn],
    )
    download_btn.click(
        fn=_download,
        inputs=[url_input, quality_dd, quality_state, dl_out_dir],
        outputs=[dl_status, video_out, audio_out, file_out, download_btn, pause_btn, stop_btn],
    )
    pause_btn.click(fn=_pause_toggle, inputs=[], outputs=[pause_btn])
    stop_btn.click(fn=_stop_download, inputs=[], outputs=[])
    dl_out_dir.change(_make_saver("dl_out_dir"), inputs=[dl_out_dir])


def build_app():
    cfg = _load_settings()          # ← load persisted settings once at startup

    with gr.Blocks(title="Lover Clinic AI Video Tools") as demo:
        gr.HTML(_header_html())

        with gr.Tabs():

            # ── AI: Photo Upscale ──────────────────────────
            with gr.Tab("🔬 AI เพิ่มความชัด (ภาพ)"):
                gr.HTML('<div class="sec-head">AI เพิ่มความคมชัดภาพ · Real-ESRGAN</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        up_in = gr.Image(label="ภาพต้นฉบับ", type="pil", height=380)
                        with gr.Row():
                            up_scale = gr.Radio(
                                choices=[2, 4], value=cfg["up_scale"], label="ขยายกี่เท่า", type="value"
                            )
                            up_model = gr.Radio(
                                choices=["General Photo", "General (Lightweight)", "Anime / Illustration"],
                                value=cfg["up_model"], label="โมเดล"
                            )
                        with gr.Row():
                            up_btn  = gr.Button("🚀 เพิ่มความชัด", variant="primary", scale=4)
                            up_stop = gr.Button("⏹ หยุด", variant="stop", scale=1, min_width=90)
                    with gr.Column(scale=1):
                        up_out = gr.Image(label="ผลลัพธ์", height=380, interactive=False)
                        up_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    up_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["up_fmt"], label="รูปแบบบันทึก", scale=1)
                up_out_dir, _ = _save_dir_row("photo")
                up_out_dir.value = cfg["up_out_dir"]
                up_event = up_btn.click(upscale_photo, inputs=[up_in, up_scale, up_model, up_out_dir, up_fmt], outputs=[up_out, up_status])
                up_stop.click(fn=None, cancels=[up_event])
                # persist on change
                up_scale.change(_make_saver("up_scale"), inputs=[up_scale])
                up_model.change(_make_saver("up_model"), inputs=[up_model])
                up_fmt.change(_make_saver("up_fmt"), inputs=[up_fmt])
                up_out_dir.change(_make_saver("up_out_dir"), inputs=[up_out_dir])

            # ── AI: Video Upscale ──────────────────────────
            with gr.Tab("🎬 AI เพิ่มความชัด (วีดีโอ)"):
                gr.HTML('<div class="sec-head">AI เพิ่มความคมชัดวีดีโอ · Real-ESRGAN (4x)</div>')
                gr.HTML(WARN_GPU)
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="วีดีโอต้นฉบับ")
                        with gr.Row():
                            vid_scale = gr.Radio(
                                choices=[2, 4], value=cfg["vid_scale"], label="ขยายกี่เท่า (General เท่านั้น)",
                                type="value"
                            )
                            vid_model = gr.Radio(
                                choices=["General (Best Quality)", "General (Fast)", "Anime / Cartoon"],
                                value=cfg["vid_model"], label="โมเดล"
                            )
                        with gr.Row():
                            vid_btn  = gr.Button("🚀 เพิ่มความชัด", variant="primary", scale=4)
                            vid_stop = gr.Button("⏹ หยุด", variant="stop", scale=1, min_width=90)
                    with gr.Column(scale=1):
                        vid_out = gr.Video(label="วีดีโอผลลัพธ์", visible=True)
                        vid_status = gr.Markdown(value="", elem_classes=["lc-status"])
                vid_out_dir, _ = _save_dir_row("video")
                vid_out_dir.value = cfg["vid_out_dir"]
                vid_event = vid_btn.click(upscale_video, inputs=[vid_in, vid_scale, vid_model, vid_out_dir], outputs=[vid_out, vid_status])
                vid_stop.click(fn=None, cancels=[vid_event])
                vid_scale.change(_make_saver("vid_scale"), inputs=[vid_scale])
                vid_model.change(_make_saver("vid_model"), inputs=[vid_model])
                vid_out_dir.change(_make_saver("vid_out_dir"), inputs=[vid_out_dir])

            # ── AI: Remove Background ──────────────────────
            with gr.Tab("✂️ AI ลบพื้นหลัง"):
                gr.HTML('<div class="sec-head">AI ลบพื้นหลัง · BiRefNet</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        bg_in = gr.Image(label="ภาพต้นฉบับ", type="pil", height=340)
                        bg_model = gr.Dropdown(
                            choices=["BiRefNet — General (Best)", "BiRefNet — Portrait", "U2Net", "RMBG 1.4"],
                            value=cfg["bg_model"], label="โมเดล AI",
                        )
                        bg_option = gr.Radio(
                            choices=["โปร่งใส", "ขาว", "ดำ", "แดง", "รูปภาพเอง"],
                            value=cfg["bg_option"], label="พื้นหลังใหม่",
                        )
                        bg_custom = gr.Image(
                            label="รูปพื้นหลังเอง", type="pil",
                            visible=(cfg["bg_option"] == "รูปภาพเอง"), height=130
                        )
                        with gr.Row():
                            bg_btn  = gr.Button("🚀 ลบพื้นหลัง", variant="primary", scale=4)
                            bg_stop = gr.Button("⏹ หยุด", variant="stop", scale=1, min_width=90)

                        def toggle_custom(choice):
                            return gr.update(visible=(choice == "รูปภาพเอง"))
                        bg_option.change(toggle_custom, bg_option, bg_custom)

                    with gr.Column(scale=1):
                        bg_out = gr.Image(label="ผลลัพธ์", height=340, interactive=False)
                        bg_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    bg_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["bg_fmt"], label="รูปแบบบันทึก (โปร่งใสต้องเป็น PNG)", scale=1)
                bg_out_dir, _ = _save_dir_row("remove_bg")
                bg_out_dir.value = cfg["bg_out_dir"]
                bg_event = bg_btn.click(remove_background, inputs=[bg_in, bg_model, bg_option, bg_custom, bg_out_dir, bg_fmt], outputs=[bg_out, bg_status])
                bg_stop.click(fn=None, cancels=[bg_event])
                bg_model.change(_make_saver("bg_model"), inputs=[bg_model])
                bg_option.change(_make_saver("bg_option"), inputs=[bg_option])
                bg_fmt.change(_make_saver("bg_fmt"), inputs=[bg_fmt])
                bg_out_dir.change(_make_saver("bg_out_dir"), inputs=[bg_out_dir])

            # ── AI: Enhance Image ──────────────────────────
            with gr.Tab("🪄 AI ฟื้นฟูภาพ"):
                gr.HTML('<div class="sec-head">AI ฟื้นฟูภาพ · GFPGAN v1.4</div>')
                gr.HTML(
                    '<div class="lc-info-box">'
                    '🪄 ฟื้นฟูรูปเก่า รูปแตก รูปเสีย — AI จะตรวจจับและซ่อมแซมใบหน้าโดยเฉพาะ<br>'
                    '✅ เหมาะกับ: รูปถ่ายเก่า · รูปพิกเซลแตก · ภาพถ่ายไม่ชัด · รูปถ่ายจากกล้องเก่า<br>'
                    '⚠️ ผลลัพธ์ดีที่สุดเมื่อรูปมีใบหน้าคน — ใช้ AI Upscale Photo สำหรับรูปที่ไม่มีหน้าคน'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        enh_in = gr.Image(label="รูปที่ต้องการฟื้นฟู", type="pil", height=340)
                        enh_scale = gr.Radio(
                            choices=[1, 2], value=cfg["enh_scale"], label="ขยายกี่เท่า", type="value"
                        )
                        enh_bg = gr.Checkbox(value=cfg["enh_bg"], label="ขยาย/ปรับภาพพื้นหลังด้วย (Real-ESRGAN)")
                        with gr.Row():
                            enh_btn  = gr.Button("🪄 ฟื้นฟูภาพ", variant="primary", scale=4)
                            enh_stop = gr.Button("⏹ หยุด", variant="stop", scale=1, min_width=90)
                    with gr.Column(scale=1):
                        enh_out = gr.Image(label="ผลลัพธ์ที่ฟื้นฟูแล้ว", height=340, interactive=False)
                        enh_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    enh_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["enh_fmt"], label="รูปแบบบันทึก", scale=1)
                enh_out_dir, _ = _save_dir_row("enhance")
                enh_out_dir.value = cfg["enh_out_dir"]
                enh_event = enh_btn.click(enhance_image, inputs=[enh_in, enh_scale, enh_bg, enh_out_dir, enh_fmt], outputs=[enh_out, enh_status])
                enh_stop.click(fn=None, cancels=[enh_event])
                enh_scale.change(_make_saver("enh_scale"), inputs=[enh_scale])
                enh_bg.change(_make_saver("enh_bg"), inputs=[enh_bg])
                enh_fmt.change(_make_saver("enh_fmt"), inputs=[enh_fmt])
                enh_out_dir.change(_make_saver("enh_out_dir"), inputs=[enh_out_dir])

            # ── TOOLS: Resize ──────────────────────────────
            with gr.Tab("📐 ปรับขนาดภาพ"):
                gr.HTML('<div class="sec-head">ปรับขนาดภาพ</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        res_in = gr.Image(label="ภาพต้นฉบับ", type="pil", height=320)
                        res_dims = gr.Textbox(label="ขนาดปัจจุบัน", interactive=False, lines=1)

                        def show_dims(img):
                            return f"{img.width} × {img.height} px" if img else ""
                        res_in.change(show_dims, res_in, res_dims)

                        with gr.Row():
                            res_w = gr.Number(label="ความกว้าง (px)", value=cfg["res_w"], minimum=1, maximum=16384)
                            res_h = gr.Number(label="ความสูง (px)", value=cfg["res_h"], minimum=1, maximum=16384)
                        res_ar = gr.Checkbox(value=cfg["res_ar"], label="คงสัดส่วนภาพ")
                        res_filter = gr.Dropdown(
                            choices=["Lanczos (คุณภาพสูงสุด)", "Bicubic", "Bilinear", "Nearest (เร็วสุด)"],
                            value=cfg["res_filter"], label="วิธีปรับขนาด"
                        )
                        res_btn = gr.Button("📐 ปรับขนาด", variant="primary")
                    with gr.Column(scale=1):
                        res_out = gr.Image(label="ผลลัพธ์", height=320, interactive=False)
                        res_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    res_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["res_fmt"], label="รูปแบบบันทึก", scale=1)
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
            with gr.Tab("✂️ ครอปภาพ"):
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
                        crop_out = gr.Image(label="ผลลัพธ์ที่ครอป", interactive=False)
                        crop_status = gr.Markdown(value="", elem_classes=["lc-status"])
                with gr.Row():
                    crop_fmt = gr.Radio(choices=["PNG", "JPEG", "WEBP"], value=cfg["crop_fmt"], label="รูปแบบบันทึก", scale=1)
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
            with gr.Tab("🔄 แปลงรูปแบบ"):
                gr.HTML('<div class="sec-head">แปลงรูปแบบไฟล์ภาพ</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        conv_in = gr.Image(label="ภาพต้นฉบับ", type="pil", height=300)
                        conv_fmt = gr.Radio(
                            choices=["JPEG", "PNG", "WEBP", "BMP", "TIFF"],
                            value=cfg["conv_fmt"], label="รูปแบบที่ต้องการ"
                        )
                        conv_q = gr.Slider(1, 100, value=cfg["conv_q"], step=1, label="คุณภาพ (เฉพาะ JPEG / WEBP)")
                        conv_btn = gr.Button("🔄 แปลงรูปแบบ", variant="primary")
                    with gr.Column(scale=1):
                        conv_preview = gr.Image(label="ตัวอย่าง", height=300, interactive=False)
                        conv_file = gr.File(label="⬇️ ดาวน์โหลดไฟล์")
                        conv_status = gr.Markdown(value="", elem_classes=["lc-status"])
                conv_out_dir, _ = _save_dir_row("convert")
                conv_out_dir.value = cfg["conv_out_dir"]
                conv_btn.click(convert_format, inputs=[conv_in, conv_fmt, conv_q, conv_out_dir], outputs=[conv_preview, conv_file, conv_status])
                conv_fmt.change(_make_saver("conv_fmt"), inputs=[conv_fmt])
                conv_q.change(_make_saver("conv_q"), inputs=[conv_q])
                conv_out_dir.change(_make_saver("conv_out_dir"), inputs=[conv_out_dir])

            # ── Download Video ─────────────────────────────
            with gr.Tab("⬇️ ดาวน์โหลดวีดีโอ"):
                _build_download_tab(cfg)

    demo.queue()
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
        css=CSS,
        js=LIGHTBOX_JS,
    )
