# pigmemento-ml

**Pigmemento ML** is a lightweight Python microservice providing melanoma image inference and CAM (Class Activation Map) heatmap generation for the Pigmemento mobile training app.

This service exposes a simple HTTP API (via **FastAPI**) that accepts dermatoscopic images and returns:

- Benign / malignant probability scores
- A CAM-like heatmap visualization
- A stable API contract for the ASP.NET Core backend

> **Note:** This service is for **educational use only** — not for diagnosis or clinical decision-making.

---

## 🚀 Features

### ✅ FastAPI microservice
- Simple & fast REST interface
- Automatic Swagger UI docs at `/docs`

### ✅ Inference stub (placeholder model)
- Deterministic, non-medical probabilities
- Easy to replace with a real PyTorch or ONNX model later

### ✅ CAM heatmap generator
- Center‑crops the original input image
- Applies a fake "heatmap" transform (placeholder)
- Saves output under `/cams/<id>.png`
- Static file serving included

### 🎯 Future‑proof architecture
Designed so you can later add:
- PyTorch inference + Grad‑CAM
- ONNX Runtime inference
- GPU acceleration
- Batch processing

---

## 📁 Project Structure

```
pigmemento-ml/
│
├── app/
│   ├── main.py          # FastAPI app entrypoint
│   ├── api.py           # /infer route
│   ├── model.py         # Stub model + image loading
│   ├── cam.py           # CAM generation helpers
│   ├── config.py        # Paths & directory config
│   ├── schemas.py       # Pydantic response models
│   └── __init__.py
│
├── cams/                # Generated CAM images (gitignored)
├── requirements.txt
└── README.md
```

---

## 🧪 API Endpoints

### **POST /infer**
Upload an image and receive probabilities + CAM URL.

#### Example response
```json
{
  "probs": {
    "benign": 0.42,
    "malignant": 0.58
  },
  "camPngUrl": "/cams/abc123.png"
}
```

#### Example request (cURL)
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

---

## 🛠️ Local Development

### 1. Clone the repo
```bash
git clone https://github.com/<youruser>/pigmemento-ml.git
cd pigmemento-ml
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Service available at:
```
http://localhost:8000
```
Docs:
```
http://localhost:8000/docs
```

---

## 🔧 Configuration
CAM images are saved to:
```
app/cams/
```
Configured via `app/config.py`:
```python
CAM_DIR = Path(__file__).resolve().parent.parent / "cams"
```
This folder is created automatically.

---

## 🟦 CAM Stub Implementation
Below is the full placeholder CAM implementation used for now:

```python
from uuid import uuid4
from PIL import Image
from .config import CAM_DIR


def center_crop(image: Image.Image) -> Image.Image:
    """Center-crop the image to a square while preserving resolution."""
    width, height = image.size
    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    return image.crop((left, top, right, bottom))


def generate_cam_stub(image: Image.Image) -> str:
    """
    Generate a fake CAM image:
    - Center-crop the original image to a square
    - Apply a simple darkening transform as a placeholder "heatmap"
    - Save it under CAM_DIR and return its URL path
    """
    # 1) Crop
    cropped = center_crop(image)

    # 2) Fake CAM transform
    heatmap = cropped.copy()
    heatmap = heatmap.point(lambda p: p * 0.7)

    # 3) Save
    CAM_DIR.mkdir(exist_ok=True)
    filename = f"{uuid4()}.png"
    path = CAM_DIR / filename
    heatmap.save(path, format="PNG")

    return f"/cams/{filename}"
```

---

## 🧱 Roadmap
### Phase 1 — Current
- Stub inference
- Stub CAM
- Full integration with Pigmemento API

### Phase 2 — ML Integration
- PyTorch → ONNX conversion
- ONNX inference pipeline
- Real Grad‑CAM / Score‑CAM

### Phase 3 — Deployment
- Dockerfile
- Container hosting
- Logging & monitoring

---

## 🩺 Disclaimer
This service is strictly for **educational purposes**.  
It must **not** be used for diagnosing, treating, or managing medical conditions.