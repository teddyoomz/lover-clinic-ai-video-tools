# 🔥 Lover Clinic AI Video Tools

AI-Powered Image & Video Processing Suite — built for **Lover Clinic** with a red / black / white design system.

---

## Features

### AI Tools
| Tab | Technology | Description |
|-----|-----------|-------------|
| **🔬 AI เพิ่มความชัด (ภาพ)** | Real-ESRGAN | Upscale photos 2× or 4× — General, General (Best Quality), Anime/Cartoon models |
| **🎬 AI เพิ่มความชัด (วีดีโอ)** | Real-ESRGAN | Frame-by-frame video upscaling with ffmpeg H.264 reassembly |
| **✂️ AI ลบพื้นหลัง** | BiRefNet / rembg | Background removal — transparent, solid color, or custom image replacement |
| **🪄 AI ฟื้นฟูภาพ** | GFPGAN | Face restoration + optional Real-ESRGAN background enhancement |

### Image Tools
| Tab | Description |
|-----|-------------|
| **📐 ปรับขนาดภาพ** | Resize to exact dimensions with aspect-ratio lock and filter selection (Lanczos, Nearest…) |
| **✂️ ครอปภาพ** | Full canvas crop editor — see details below |
| **🔄 แปลงรูปแบบ** | Convert between JPEG, PNG, WEBP, BMP, TIFF with quality control |
| **⬇️ ดาวน์โหลดวีดีโอ** | Download video from YouTube, Facebook, Instagram, TikTok and 1000+ sites |

---

### Canvas Crop Editor — ✂️ ครอปภาพ

A fully interactive canvas-based crop tool:

| Feature | Details |
|---------|---------|
| **Aspect Ratio lock** | Free, 1:1, 4:3, 3:4, 16:9, 9:16, 4:5, 5:4, 3:2, 2:3 + Swap / Center / Clear |
| **Social presets** | IG Feed, Square, Story/Reel, YouTube, FB Cover, LinkedIn, Pinterest, OG Image |
| **Rotation** | 90° Left / 90° Right / 180° |
| **Flip** | Horizontal / Vertical + full reset |
| **Grid overlays** | Rule of Thirds, Golden Ratio (φ), Diagonal+Cross — both on crop box and full image |
| **Zoom** | 25% → 300% with persistent zoom level (remembers across sessions) |
| **Touch support** | Full mobile drag / resize / move via touch events |
| **Output** | Crops at original resolution — canvas is just a preview at display scale |

---

## Installation (via Pinokio)

1. Open **Lover Clinic AI Video Tools** in Pinokio.
2. Click **Install** — this will:
   - Install `ffmpeg` via conda
   - Install all Python dependencies (Real-ESRGAN, GFPGAN, rembg, Gradio…)
   - Auto-detect GPU (NVIDIA CUDA / AMD / Apple MPS / CPU) and install the correct PyTorch
3. Click **Start** — the web UI opens automatically in your browser.

> **First run:** AI models are downloaded automatically on first use. Allow a few minutes per tool.

---

## Usage

After starting, the web UI opens at `http://127.0.0.1:7860`.

- All output folders are configurable per-tool and remembered between sessions.
- Click **📂 เปิดโฟลเดอร์** to open the output folder in Explorer.
- GPU info is shown in the top-right badge.

---

## API

The app exposes a standard Gradio API at `http://127.0.0.1:7860`.

### JavaScript (fetch)
```javascript
const res = await fetch("http://127.0.0.1:7860/run/upscale_photo", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ data: ["data:image/jpeg;base64,<BASE64>", 4, "General Photo", "", "PNG"] })
})
const { data } = await res.json()
// data[0] = upscaled image,  data[1] = status message
```

### Python (gradio_client)
```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

# Upscale photo
result = client.predict(
    image="path/to/photo.jpg",
    scale=4,
    model_type="General Photo",
    api_name="/upscale_photo"
)

# Remove background
result = client.predict(
    image="path/to/photo.jpg",
    model_choice="BiRefNet — General (Best)",
    bg_option="โปร่งใส",
    custom_bg=None,
    api_name="/remove_background"
)
```

### cURL
```bash
curl -X POST http://127.0.0.1:7860/run/upscale_photo \
  -H "Content-Type: application/json" \
  -d '{"data": ["data:image/jpeg;base64,<BASE64>", 4, "General Photo", "", "PNG"]}'
```

---

## Technologies

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** — BSD-3-Clause
- **[GFPGAN](https://github.com/TencentARC/GFPGAN)** — Apache 2.0
- **[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)** via **[rembg](https://github.com/danielgatis/rembg)** — MIT
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — Unlicense
- **[Gradio](https://gradio.app)** — Apache 2.0

---

*🔥 Lover Clinic AI Video Tools — Powered by open-source AI*
