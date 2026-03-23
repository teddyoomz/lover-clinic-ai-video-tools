import gradio as gr
import os
import sys
import tempfile
import shutil
import subprocess
import logging
import traceback
from pathlib import Path
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
# ============================================================

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    try:
        import torchvision.transforms.functional as _ft
        sys.modules["torchvision.transforms.functional_tensor"] = _ft
        logger.info("Applied torchvision.functional_tensor compatibility patch")
    except Exception as _e:
        logger.warning(f"Could not apply torchvision patch: {_e}")

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
    except Exception:
        pass
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

def upscale_photo(image, scale, model_type, progress=gr.Progress()):
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
        return result, f"✅ Upscaled {scale}x — {result.width}×{result.height} px"
    except Exception as e:
        logger.error(f"upscale_photo failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def upscale_video(video_path, scale, model_type, progress=gr.Progress()):
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
        out_video = os.path.join(tmpdir, "output.mp4")

        # Try ffmpeg first, fall back to cv2
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(out_dir, "frame_%08d.png"),
                "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p",
                out_video
            ], check=True, capture_output=True)
        except Exception:
            sample = cv2.imread(out_paths[0])
            h, w = sample.shape[:2]
            writer = cv2.VideoWriter(
                out_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h)
            )
            for op in out_paths:
                writer.write(cv2.imread(op))
            writer.release()

        # Copy result so tmpdir can be cleaned safely
        final_path = os.path.join(tempfile.gettempdir(), "lc_upscaled_video.mp4")
        shutil.copy2(out_video, final_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

        progress(1.0, desc="Done!")
        return final_path, f"✅ Video upscaled {scale}x successfully!"
    except Exception as e:
        logger.error(f"upscale_video failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def remove_background(image, model_choice, bg_option, custom_bg,
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
        return final, "✅ Background removed successfully!"
    except Exception as e:
        logger.error(f"remove_background failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def enhance_image(image, upscale_factor, enhance_bg, progress=gr.Progress()):
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
        return result, "✅ Image enhanced successfully!"
    except Exception as e:
        logger.error(f"enhance_image failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


# ============================================================
# TOOLS FUNCTIONS
# ============================================================

def resize_image(image, width, height, maintain_ar, resample_filter):
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

        return result, f"✅ Resized to {result.width}×{result.height} px"
    except Exception as e:
        logger.error(f"resize_image failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def crop_image(image, left, top, right, bottom):
    if image is None:
        return None, "⚠️ Please upload an image first."
    try:
        iw, ih = image.size
        l = max(0, min(int(left), iw - 1))
        t = max(0, min(int(top), ih - 1))
        r = max(l + 1, min(int(right), iw))
        b = max(t + 1, min(int(bottom), ih))
        result = image.crop((l, t, r, b))
        return result, f"✅ Cropped to {result.width}×{result.height} px"
    except Exception as e:
        logger.error(f"crop_image failed: {e}\n{traceback.format_exc()}")
        return None, f"❌ Error: {e}"


def convert_format(image, out_format, quality):
    if image is None:
        return None, None, "⚠️ Please upload an image first."
    try:
        ext_map = {
            "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
            "BMP": ".bmp", "TIFF": ".tiff",
        }
        ext = ext_map.get(out_format, ".jpg")
        out_path = os.path.join(tempfile.gettempdir(), f"lc_converted{ext}")

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
        return preview, out_path, f"✅ Converted to {out_format} — saved as lc_converted{ext}"
    except Exception as e:
        logger.error(f"convert_format failed: {e}\n{traceback.format_exc()}")
        return None, None, f"❌ Error: {e}"


# ============================================================
# THEME & CSS
# ============================================================

CSS = """
/* ======= Lover Clinic — Red / Black / White / Fire Theme ======= */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --lc-red:        #CC0000;
  --lc-red-bright: #FF1111;
  --lc-fire:       #FF4500;
  --lc-fire2:      #FF7700;
  --lc-dark:       #080808;
  --lc-dark2:      #101010;
  --lc-surface:    #181818;
  --lc-surface2:   #202020;
  --lc-border:     #2A0000;
  --lc-border2:    #440000;
  --lc-white:      #F2F2F2;
  --lc-muted:      #777777;
  --radius:        8px;
}

/* ---- Base ---- */
body, .gradio-container {
  background: var(--lc-dark) !important;
  color: var(--lc-white) !important;
  font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1440px !important; }
footer { display: none !important; }

/* ---- Header ---- */
#lc-header {
  background: linear-gradient(135deg, #0A0000 0%, #1A0000 40%, #0A0000 100%);
  border-bottom: 2px solid var(--lc-red);
  padding: 18px 28px 14px;
  display: flex; align-items: center; gap: 18px;
  margin-bottom: 4px;
}
#lc-header h1 {
  font-family: 'Rajdhani', sans-serif;
  font-size: 26px; font-weight: 700;
  color: #fff; margin: 0;
  text-transform: uppercase; letter-spacing: 3px;
  text-shadow: 0 0 24px rgba(255,60,0,0.5);
}
#lc-header .sub {
  font-size: 11px; color: var(--lc-muted);
  letter-spacing: 1.5px; text-transform: uppercase; margin-top: 3px;
}

/* ---- Tabs ---- */
.tab-nav { background: var(--lc-dark2) !important; border-bottom: 1px solid var(--lc-border2) !important; }
.tab-nav button {
  color: var(--lc-muted) !important; background: transparent !important;
  border: none !important; font-size: 13px !important;
  padding: 11px 18px !important; letter-spacing: 0.4px !important;
  transition: all 0.2s !important;
}
.tab-nav button:hover { color: var(--lc-white) !important; background: rgba(204,0,0,0.08) !important; }
.tab-nav button.selected {
  color: #fff !important;
  background: linear-gradient(180deg, #990000 0%, var(--lc-red) 100%) !important;
  border-bottom: 2px solid var(--lc-fire) !important;
}

/* ---- Blocks / Panels ---- */
.block, .panel {
  background: var(--lc-surface) !important;
  border: 1px solid var(--lc-border) !important;
  border-radius: var(--radius) !important;
}
.gap { gap: 12px !important; }

/* ---- Section headings ---- */
.sec-head {
  font-family: 'Rajdhani', sans-serif;
  font-size: 16px; font-weight: 700; letter-spacing: 2px;
  color: var(--lc-red); text-transform: uppercase;
  border-bottom: 1px solid var(--lc-border2);
  padding-bottom: 8px; margin-bottom: 14px;
}

/* ---- Labels ---- */
label span, .label-wrap span {
  color: var(--lc-muted) !important;
  font-size: 11px !important; font-weight: 500 !important;
  letter-spacing: 0.6px !important; text-transform: uppercase !important;
}

/* ---- Inputs ---- */
input[type="text"], input[type="number"], textarea {
  background: var(--lc-surface2) !important;
  border: 1px solid var(--lc-border2) !important;
  color: var(--lc-white) !important;
  border-radius: 6px !important;
}
input:focus, textarea:focus {
  border-color: var(--lc-red) !important;
  box-shadow: 0 0 0 2px rgba(204,0,0,0.25) !important;
  outline: none !important;
}

/* ---- Buttons ---- */
button.primary, .btn-primary {
  background: linear-gradient(135deg, #990000, var(--lc-fire)) !important;
  border: none !important; color: #fff !important;
  font-weight: 600 !important; letter-spacing: 1px !important;
  text-transform: uppercase !important; border-radius: var(--radius) !important;
  padding: 11px 22px !important; transition: all 0.2s !important;
  box-shadow: 0 4px 18px rgba(204,0,0,0.35) !important;
}
button.primary:hover {
  background: linear-gradient(135deg, var(--lc-red-bright), var(--lc-fire2)) !important;
  box-shadow: 0 6px 24px rgba(255,17,17,0.45) !important;
  transform: translateY(-1px) !important;
}
button.secondary {
  background: var(--lc-surface2) !important;
  border: 1px solid var(--lc-border2) !important;
  color: var(--lc-white) !important; border-radius: var(--radius) !important;
}

/* ---- Upload zone ---- */
.upload-button, [data-testid="image"] .wrap {
  background: var(--lc-surface2) !important;
  border: 2px dashed var(--lc-border2) !important;
  border-radius: var(--radius) !important;
}
.upload-button:hover, [data-testid="image"] .wrap:hover {
  border-color: var(--lc-red) !important;
  background: rgba(204,0,0,0.05) !important;
}

/* ---- Sliders ---- */
input[type="range"] { accent-color: var(--lc-red) !important; }

/* ---- Radio / Checkbox ---- */
input[type="radio"], input[type="checkbox"] { accent-color: var(--lc-red) !important; }

/* ---- Dropdown ---- */
.wrap-inner, select {
  background: var(--lc-surface2) !important;
  border-color: var(--lc-border2) !important;
  color: var(--lc-white) !important;
}

/* ---- Info text / status ---- */
.info-box {
  background: #1A0A0A; border-left: 3px solid var(--lc-fire);
  border-radius: 0 6px 6px 0; padding: 9px 14px;
  font-size: 12px; color: #FF9966; margin-bottom: 12px;
}

/* ---- Fire divider ---- */
.fire-line {
  height: 2px; border: none; margin: 12px 0;
  background: linear-gradient(90deg, transparent, var(--lc-red), var(--lc-fire), var(--lc-red), transparent);
}
"""

HEADER_HTML = """
<div id="lc-header">
  <div>
    <h1>🔥 Lover Clinic AI Video Tools</h1>
    <div class="sub">AI-Powered Image &amp; Video Processing Suite — Upscale · Enhance · Remove BG · Convert</div>
  </div>
</div>
"""

WARN_GPU = """<div class="info-box">
⚠️ Video upscaling is GPU-intensive. Processing may take several minutes per minute of video.
GPU strongly recommended.
</div>"""


# ============================================================
# GRADIO UI
# ============================================================

def build_app():
    with gr.Blocks(title="Lover Clinic AI Video Tools") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── AI: Photo Upscale ──────────────────────────
            with gr.Tab("🔬 AI Upscale Photo"):
                gr.HTML('<div class="sec-head">AI Photo Upscaler · Real-ESRGAN</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        up_in = gr.Image(label="Input Image", type="pil", height=380)
                        with gr.Row():
                            up_scale = gr.Radio(
                                choices=[2, 4], value=4, label="Scale Factor", type="value"
                            )
                            up_model = gr.Radio(
                                choices=["General Photo", "Anime / Illustration"],
                                value="General Photo", label="Model"
                            )
                        up_btn = gr.Button("🚀 Upscale Photo", variant="primary")
                    with gr.Column(scale=1):
                        up_out = gr.Image(label="Upscaled Result", height=380)
                        up_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                up_btn.click(
                    upscale_photo,
                    inputs=[up_in, up_scale, up_model],
                    outputs=[up_out, up_status],
                )

            # ── AI: Video Upscale ──────────────────────────
            with gr.Tab("🎬 AI Upscale Video"):
                gr.HTML('<div class="sec-head">AI Video Upscaler · Real-ESRGAN</div>')
                gr.HTML(WARN_GPU)
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Input Video")
                        with gr.Row():
                            vid_scale = gr.Radio(
                                choices=[2, 4], value=2, label="Scale Factor", type="value"
                            )
                            vid_model = gr.Radio(
                                choices=["General", "Anime"],
                                value="General", label="Model"
                            )
                        vid_btn = gr.Button("🚀 Upscale Video", variant="primary")
                    with gr.Column(scale=1):
                        vid_out = gr.Video(label="Upscaled Video")
                        vid_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                vid_btn.click(
                    upscale_video,
                    inputs=[vid_in, vid_scale, vid_model],
                    outputs=[vid_out, vid_status],
                )

            # ── AI: Remove Background ──────────────────────
            with gr.Tab("✂️ AI Remove BG"):
                gr.HTML('<div class="sec-head">AI Background Remover · BiRefNet</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        bg_in = gr.Image(label="Input Image", type="pil", height=340)
                        bg_model = gr.Dropdown(
                            choices=[
                                "BiRefNet — General (Best)",
                                "BiRefNet — Portrait",
                                "U2Net",
                                "RMBG 1.4",
                            ],
                            value="BiRefNet — General (Best)",
                            label="AI Model",
                        )
                        bg_option = gr.Radio(
                            choices=["Transparent", "White", "Black", "Red", "Custom Image"],
                            value="Transparent",
                            label="Background Replacement",
                        )
                        bg_custom = gr.Image(
                            label="Custom Background Image", type="pil",
                            visible=False, height=130
                        )
                        bg_btn = gr.Button("🚀 Remove Background", variant="primary")

                        def toggle_custom(choice):
                            return gr.update(visible=(choice == "Custom Image"))
                        bg_option.change(toggle_custom, bg_option, bg_custom)

                    with gr.Column(scale=1):
                        bg_out = gr.Image(label="Result", height=340)
                        bg_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                bg_btn.click(
                    remove_background,
                    inputs=[bg_in, bg_model, bg_option, bg_custom],
                    outputs=[bg_out, bg_status],
                )

            # ── AI: Enhance Image ──────────────────────────
            with gr.Tab("✨ AI Enhance Image"):
                gr.HTML('<div class="sec-head">AI Image Enhancer · GFPGAN</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        enh_in = gr.Image(label="Input Image", type="pil", height=340)
                        enh_scale = gr.Radio(
                            choices=[1, 2], value=2, label="Upscale Factor", type="value"
                        )
                        enh_bg = gr.Checkbox(value=True, label="Also enhance background (Real-ESRGAN)")
                        enh_btn = gr.Button("🚀 Enhance Image", variant="primary")
                    with gr.Column(scale=1):
                        enh_out = gr.Image(label="Enhanced Result", height=340)
                        enh_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                enh_btn.click(
                    enhance_image,
                    inputs=[enh_in, enh_scale, enh_bg],
                    outputs=[enh_out, enh_status],
                )

            # ── TOOLS: Resize ──────────────────────────────
            with gr.Tab("📐 Resize Image"):
                gr.HTML('<div class="sec-head">Image Resizer</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        res_in = gr.Image(label="Input Image", type="pil", height=320)
                        res_dims = gr.Textbox(
                            label="Current Dimensions", interactive=False, lines=1
                        )

                        def show_dims(img):
                            return f"{img.width} × {img.height} px" if img else ""
                        res_in.change(show_dims, res_in, res_dims)

                        with gr.Row():
                            res_w = gr.Number(label="Width (px)", value=1920, minimum=1, maximum=16384)
                            res_h = gr.Number(label="Height (px)", value=1080, minimum=1, maximum=16384)
                        res_ar = gr.Checkbox(value=True, label="Maintain Aspect Ratio")
                        res_filter = gr.Dropdown(
                            choices=["Lanczos (Best Quality)", "Bicubic", "Bilinear", "Nearest (Fastest)"],
                            value="Lanczos (Best Quality)", label="Resample Filter"
                        )
                        res_btn = gr.Button("📐 Resize", variant="primary")
                    with gr.Column(scale=1):
                        res_out = gr.Image(label="Resized Image", height=320)
                        res_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                res_btn.click(
                    resize_image,
                    inputs=[res_in, res_w, res_h, res_ar, res_filter],
                    outputs=[res_out, res_status],
                )

            # ── TOOLS: Crop ────────────────────────────────
            with gr.Tab("✂️ Crop Image"):
                gr.HTML('<div class="sec-head">Image Cropper</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        crop_in = gr.Image(label="Input Image", type="pil", height=320)
                        crop_dims = gr.Textbox(
                            label="Image Dimensions", interactive=False, lines=1
                        )

                        def show_crop_info(img):
                            return f"Width: {img.width}px  Height: {img.height}px" if img else ""
                        crop_in.change(show_crop_info, crop_in, crop_dims)

                        with gr.Row():
                            crop_left = gr.Number(label="Left (px)", value=0, minimum=0)
                            crop_top = gr.Number(label="Top (px)", value=0, minimum=0)
                        with gr.Row():
                            crop_right = gr.Number(label="Right (px)", value=1920, minimum=1)
                            crop_bottom = gr.Number(label="Bottom (px)", value=1080, minimum=1)

                        def auto_fill(img):
                            if img:
                                return 0, 0, img.width, img.height
                            return 0, 0, 1920, 1080
                        crop_in.change(
                            auto_fill, crop_in,
                            [crop_left, crop_top, crop_right, crop_bottom]
                        )

                        crop_btn = gr.Button("✂️ Crop", variant="primary")
                    with gr.Column(scale=1):
                        crop_out = gr.Image(label="Cropped Image", height=320)
                        crop_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                crop_btn.click(
                    crop_image,
                    inputs=[crop_in, crop_left, crop_top, crop_right, crop_bottom],
                    outputs=[crop_out, crop_status],
                )

            # ── TOOLS: Convert ─────────────────────────────
            with gr.Tab("🔄 Convert Format"):
                gr.HTML('<div class="sec-head">Image Format Converter</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        conv_in = gr.Image(label="Input Image", type="pil", height=300)
                        conv_fmt = gr.Radio(
                            choices=["JPEG", "PNG", "WEBP", "BMP", "TIFF"],
                            value="WEBP", label="Output Format"
                        )
                        conv_q = gr.Slider(
                            1, 100, value=95, step=1,
                            label="Quality (JPEG / WEBP only)"
                        )
                        conv_btn = gr.Button("🔄 Convert", variant="primary")
                    with gr.Column(scale=1):
                        conv_preview = gr.Image(label="Preview", height=300)
                        conv_file = gr.File(label="⬇️ Download Converted File")
                        conv_status = gr.Textbox(show_label=False, interactive=False, lines=1)

                conv_btn.click(
                    convert_format,
                    inputs=[conv_in, conv_fmt, conv_q],
                    outputs=[conv_preview, conv_file, conv_status],
                )

        gr.HTML("""
        <div style="text-align:center;padding:16px 0 8px;
                    color:#444;font-size:11px;
                    border-top:1px solid #2A0000;margin-top:16px;
                    letter-spacing:1px;">
          🔥 LOVER CLINIC AI VIDEO TOOLS &nbsp;·&nbsp;
          Real-ESRGAN &nbsp;·&nbsp; GFPGAN &nbsp;·&nbsp; BiRefNet
        </div>
        """)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app()
    logger.info(f"Launching on http://127.0.0.1:{args.port}")
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=True,
        favicon_path=None,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.red,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
