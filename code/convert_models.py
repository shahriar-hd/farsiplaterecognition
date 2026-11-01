"""
Conversion script for Iranian License Plate Recognition (YOLOv8) models.
Exports:
1. Safetensors (for Hugging Face standard weight format)
2. ONNX (640x640 optimized for browser inference via onnxruntime-web)
3. NCNN (640x640 and 320x320 for Raspberry Pi / embedded edge inference)
Prepares all release files into `.uploads/` directory ready for Hugging Face Hub upload.
"""

import os
import shutil
import json
import yaml
import torch
from safetensors.torch import save_file
from ultralytics import YOLO

CHECKPOINT_PATH = "checkpoints/iran_lpr_model19/weights/best.pt"
OUTPUT_DIR = ".uploads"

CLASSES = [
    "plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "alef", "be", "pe", "te", "se", "jim", "dal", "sin", "shin",
    "sad", "ta", "za", "ein", "fe", "ghaf", "kaf", "gaf", "lam",
    "mim", "noon", "vav", "he", "ye", "ze", "taxi", "disabled",
    "police", "D", "S", "protocol"
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def convert_to_safetensors(model_pt_path, output_dir):
    print("\n--- [1/4] Converting to SafeTensors ---")
    ckpt = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"].state_dict()
    else:
        state_dict = ckpt

    # Filter/clean tensors for safetensors compatibility
    clean_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean_state_dict[k] = v.contiguous()

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(clean_state_dict, safetensors_path, metadata={"format": "pt", "task": "detect"})
    print(f"✅ SafeTensors saved to: {safetensors_path} ({os.path.getsize(safetensors_path) / (1024*1024):.2f} MB)")

def convert_to_onnx(model_pt_path, output_dir):
    print("\n--- [2/4] Converting to ONNX (640x640) ---")
    model = YOLO(model_pt_path)
    # Export ONNX with imgsz=640 and simplification
    onnx_file = model.export(format="onnx", imgsz=640, simplify=True, opset=18)
    target_onnx = os.path.join(output_dir, "iran_lpr_yolov8_640.onnx")
    shutil.copy2(onnx_file, target_onnx)
    print(f"✅ ONNX model saved to: {target_onnx} ({os.path.getsize(target_onnx) / (1024*1024):.2f} MB)")

def prepare_ncnn_models(output_dir):
    print("\n--- [3/4] Packaging NCNN models (640 & 320 for Raspberry Pi) ---")
    ncnn_640_src = "checkpoints/best_ncnn_model_640"
    ncnn_320_src = "checkpoints/best_ncnn_model_320"

    ncnn_640_dst = os.path.join(output_dir, "ncnn_640")
    ncnn_320_dst = os.path.join(output_dir, "ncnn_320")

    ensure_dir(ncnn_640_dst)
    ensure_dir(ncnn_320_dst)

    for item in os.listdir(ncnn_640_src):
        if not item.startswith("__pycache__"):
            shutil.copy2(os.path.join(ncnn_640_src, item), os.path.join(ncnn_640_dst, item))

    for item in os.listdir(ncnn_320_src):
        if not item.startswith("__pycache__"):
            shutil.copy2(os.path.join(ncnn_320_src, item), os.path.join(ncnn_320_dst, item))

    # Also create zip archives for convenient RPi downloads
    shutil.make_archive(os.path.join(output_dir, "iran_lpr_ncnn_640"), 'zip', ncnn_640_dst)
    shutil.make_archive(os.path.join(output_dir, "iran_lpr_ncnn_320"), 'zip', ncnn_320_dst)
    print(f"✅ NCNN 640 & 320 models packaged to: {ncnn_640_dst}, {ncnn_320_dst} and zip archives")

def copy_pt_and_metadata(model_pt_path, output_dir):
    print("\n--- [4/4] Copying PyTorch best.pt and metadata ---")
    target_pt = os.path.join(output_dir, "best.pt")
    shutil.copy2(model_pt_path, target_pt)

    # config / metadata
    metadata = {
        "task": "detect",
        "model_type": "yolov8n",
        "num_classes": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
        "imgsz": 640,
        "input_tensor": "images (1, 3, 640, 640)",
        "output_tensor": "output0 (1, 45, 8400)",
        "license": "MIT"
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2, ensure_ascii=False)

    # Hugging Face Model Card README.md
    model_card = """---
language:
- fa
- en
license: mit
tags:
- yolo
- yolov8
- object-detection
- license-plate-recognition
- alpr
- anpr
- iranian-license-plates
- ncnn
- onnx
- edge-ai
- raspberry-pi
metrics:
- mAP
---

# 🚗 Iranian License Plate Recognition (YOLOv8 ALPR)

High accuracy, lightweight YOLOv8n object detection model trained to detect Iranian vehicle license plates and read Persian digits and characters in real time.

## 📦 Provided Model Formats
- `best.pt`: Original PyTorch model weights (YOLOv8).
- `model.safetensors`: SafeTensors weights for Hugging Face Hub ecosystem.
- `iran_lpr_yolov8_640.onnx`: Optimized ONNX model (640x640) for browser (ONNX Runtime Web / WebAssembly) and desktop applications.
- `ncnn_640/` & `iran_lpr_ncnn_640.zip`: Optimized NCNN model for Raspberry Pi / embedded edge devices (640x640).
- `ncnn_320/` & `iran_lpr_ncnn_320.zip`: Ultra-fast NCNN model for Raspberry Pi 3/4/5 (320x320).

## 🏷️ Classes (41 Total)
0. `plate`: Vehicle license plate bounding box
1-10. Digits: `0` through `9`
11-35. Persian Letters: `alef`, `be`, `pe`, `te`, `se`, `jim`, `dal`, `sin`, `shin`, `sad`, `ta`, `za`, `ein`, `fe`, `ghaf`, `kaf`, `gaf`, `lam`, `mim`, `noon`, `vav`, `he`, `ye`, `ze`
36-40. Special markers: `taxi`, `disabled`, `police`, `D`, `S`, `protocol`

## 🚀 Usage

### Python (Ultralytics)
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("car.jpg")
results[0].show()
```

### ONNX Runtime
```python
import onnxruntime as ort
session = ort.InferenceSession("iran_lpr_yolov8_640.onnx")
```

### NCNN (Raspberry Pi)
```bash
unzip iran_lpr_ncnn_640.zip -d ncnn_model
```
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)

    print("✅ Model card (README.md) and configs written.")

def main():
    ensure_dir(OUTPUT_DIR)
    convert_to_safetensors(CHECKPOINT_PATH, OUTPUT_DIR)
    convert_to_onnx(CHECKPOINT_PATH, OUTPUT_DIR)
    prepare_ncnn_models(OUTPUT_DIR)
    copy_pt_and_metadata(CHECKPOINT_PATH, OUTPUT_DIR)
    print("\n🎉 All models converted & prepared in .uploads/ directory successfully!")

if __name__ == "__main__":
    main()
