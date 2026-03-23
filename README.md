# 🔥 Lover Clinic AI Video Tools

AI-Powered Image & Video Processing Suite — built for **Lover Clinic** with a red / black / white / fire theme.

---

## Features

### AI Tools
| Tool | Technology | Description |
|------|-----------|-------------|
| **AI Photo Upscaler** | Real-ESRGAN | Upscale photos 2× or 4× (general + anime models) |
| **AI Video Upscaler** | Real-ESRGAN | Frame-by-frame video upscaling with ffmpeg reassembly |
| **AI Background Remover** | BiRefNet / rembg | State-of-the-art background removal with optional replacement |
| **AI Image Enhancer** | GFPGAN | Face restoration + background enhancement |

### Image Tools
| Tool | Description |
|------|-------------|
| **Resize Image** | Resize to exact dimensions or constrain by aspect ratio |
| **Crop Image** | Pixel-precise cropping with coordinate input |
| **Convert Format** | Convert between JPEG, PNG, WEBP, BMP, TIFF |

---

## Installation (via Pinokio)

1. Open the **Lover Clinic AI Video Tools** app in Pinokio.
2. Click **Install** — this will:
   - Install `ffmpeg` via conda
   - Install all Python dependencies (Real-ESRGAN, GFPGAN, rembg, Gradio…)
   - Install PyTorch for your hardware (NVIDIA CUDA, AMD, Apple Silicon, or CPU)
3. Click **Start** to launch the web UI.

---

## Usage

After starting, Pinokio opens the web UI automatically at `http://127.0.0.1:7860`.

Select the tab for the tool you need, upload your image or video, adjust settings, and click the action button.

> **Note:** AI models are downloaded automatically on first use. The first run of each tool may take a few minutes to download model weights.

---

## API

The app exposes a standard Gradio API accessible at `http://127.0.0.1:7860`.

### JavaScript (fetch)
```javascript
// Example: Upscale a photo
const formData = new FormData()
formData.append("data", JSON.stringify([imageBase64, 4, "General Photo"]))

const res = await fetch("http://127.0.0.1:7860/run/upscale_photo", {
  method: "POST",
  body: formData,
})
const { data } = await res.json()
console.log(data) // [upscaled_image, status_message]
```

### Python
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
    bg_option="Transparent",
    custom_bg=None,
    api_name="/remove_background"
)
```

### cURL
```bash
curl -X POST http://127.0.0.1:7860/run/upscale_photo \
  -H "Content-Type: application/json" \
  -d '{"data": ["data:image/jpeg;base64,<BASE64>", 4, "General Photo"]}'
```

---

## Technologies

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** — BSD-3-Clause
- **[GFPGAN](https://github.com/TencentARC/GFPGAN)** — Apache 2.0
- **[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)** via **[rembg](https://github.com/danielgatis/rembg)** — MIT
- **[Gradio](https://gradio.app)** — Apache 2.0

---

*🔥 Lover Clinic AI Video Tools — Powered by open-source AI*
