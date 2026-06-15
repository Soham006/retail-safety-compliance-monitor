# Retail Safety & Compliance AI Monitor

A real-time computer vision system that monitors face mask compliance using deep learning — applicable to retail stores, workplaces, and public spaces requiring health/safety protocol enforcement.

## 🎯 Business Context

Compliance monitoring (safety protocols, PPE usage, dress codes) is a recurring operational challenge for retail and workplace environments. Manual monitoring is costly and inconsistent. This project demonstrates an automated, camera-based compliance detection system that could be deployed at entry points or monitoring stations.

## 🧠 How It Works

- A deep learning model (MobileNetV2, via transfer learning) classifies faces as "Mask" or "No Mask" in real time
- OpenCV handles face detection and live webcam video processing
- Two deployment variants are included:
  - **Standard (Keras/.h5)** — full-precision model
  - **Optimized (TFLite)** — converted for faster inference, suitable for edge devices (e.g., in-store cameras with limited compute)

## 🛠️ Tech Stack

- Python, TensorFlow / Keras
- OpenCV (face detection, video processing)
- MobileNetV2 (transfer learning base)
- TensorFlow Lite (model optimization for deployment)

## 📁 Project Files

- `train_model.py` — trains the MobileNetV2-based classifier
- `detect_mask.py` — real-time detection using the full Keras model
- `detect_mask_tflite.py` — real-time detection using the optimized TFLite model
- `convert_model.py` — converts the trained model to TFLite format

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Soham006/retail-safety-compliance-monitor.git
cd retail-safety-compliance-monitor

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run detection (requires a webcam)
python detect_mask.py
```

**Note:** Trained model files (`.h5`) are excluded from this repository due to file size limits. Run `train_model.py` to regenerate the model (requires a labeled image dataset of masked/unmasked faces), or contact the author for the pre-trained model file.

## 👤 About

Built by **Soham Roy**, MBA candidate (Marketing & Analytics) at IMI Kolkata, with a B.Tech in Computer Science — exploring applications of AI/computer vision in retail operations and compliance monitoring.
