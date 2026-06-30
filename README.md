# Real-Time ASL Sign Language Recognition (RT-SLR-LDL)

A lightweight, edge-optimized machine learning pipeline for **real-time American Sign Language (ASL) recognition** using **MediaPipe Hands** and **TensorFlow/Keras**.

The system recognizes **36 ASL classes** consisting of:

- Digits **0–9**
- Letters **A–Z**

Instead of training directly on images, the pipeline extracts **3D hand landmarks**, normalizes them into a consistent representation, stores them in a **SQLite database**, and trains a compact neural network for fast and accurate inference.

---

# Features

-  Real-time ASL recognition
-  MediaPipe-based 3D hand landmark extraction
-  Wrist-centered geometric normalization
-  SQLite dataset management
-  Lightweight neural network (61,412 parameters)
-  Edge-device friendly
-  TensorFlow Lite export support
-  Webcam inference
-  Session logging
-  Model registry for experiment tracking

---

# Performance

The baseline model satisfies all targeted non-functional requirements.

| Metric | Target | Baseline Result | Status |
|---------|---------|----------------|--------|
| **Top-1 Validation Accuracy** | ≥ 90% | **95.32%** | ✅ Passed |
| **Model Size** | ≤ 25 MB | **0.234 MB** | ✅ Passed |
| **CPU Inference Latency** | ≤ 100 ms | **< 5 ms** | ✅ Passed |
| **Trainable Parameters** | — | **61,412** | ✅ |

---

# System Architecture

```
           Massey ASL Dataset (.jpeg)
                     │
                     ▼
             MediaPipe Hands
      (21 Hand Landmarks × X,Y,Z)
                     │
                     ▼
         Landmark Normalization
   (Translation + Scaling + Flattening)
                     │
                     ▼
             SQLite Database
      (Structured Feature Storage)
                     │
                     ▼
          TensorFlow/Keras Model
                     │
                     ▼
         TensorFlow Lite Conversion
                     │
                     ▼
      Real-Time Webcam Recognition
```

---

# Project Structure

```
RT_SLR/
│
├── workspace/
│   ├── data/
│   │   └── asl_label_map.json
│   │
│   ├── logs/
│   │   └── session_log/
│   │
│   ├── models/
│   │   ├── asl_classifier_v1.h5
│   │   └── asl_classifier_v1.tflite
│   │
│   └── asl_slr.db
│
├── ingest_dataset.py
├── train_model.py
├── realtime_demo.py
├── requirements.txt
└── README.md
```

---

# Database Schema

The pipeline stores all extracted features and experiment metadata in a relational SQLite database.

## gesture_classes

Stores the ASL class information.

| Column | Description |
|----------|-------------|
| id | Numeric class ID |
| label | ASL character |
| language | Language metadata |

---

## landmark_samples

Stores normalized landmark vectors.

| Column | Description |
|----------|-------------|
| sample_id | Primary key |
| class_id | Gesture label |
| split | Train / Validation / Test |
| landmarks | Serialized 63-value feature vector |

---

## session_log

Stores predictions generated during webcam inference.

| Column | Description |
|----------|-------------|
| timestamp | Prediction time |
| prediction | Predicted gesture |
| confidence | Model confidence |
| inference_time | Runtime latency |

---

## model_registry

Tracks trained models.

| Column | Description |
|----------|-------------|
| model_name | Saved model |
| accuracy | Validation accuracy |
| model_path | File path |
| created_at | Timestamp |

---

# Installation

## Prerequisites

- Python 3.12+
- Webcam (for live recognition)

---

## Clone Repository

```bash
git clone https://github.com/yourusername/RT-SLR-LDL.git

cd RT-SLR-LDL
```

---

## Install Dependencies

```bash
pip install tensorflow==2.19.1 \
mediapipe \
opencv-python \
numpy \
scikit-learn \
matplotlib \
pandas \
tqdm
```

or

```bash
pip install -r requirements.txt
```

---

# Data Processing Pipeline

Each image undergoes the following preprocessing stages before training.

## 1. Hand Detection

MediaPipe Hands detects **21 hand landmarks**.

```
21 landmarks
×
3 coordinates (x, y, z)
=
63 features
```

---

## 2. Translation

The wrist landmark (Landmark 0) becomes the local origin.

```
P' = P - Wrist
```

This removes dependence on the hand's absolute position within the image.

---

## 3. Scaling

All coordinates are divided by the maximum Euclidean distance from the wrist.

```
P_scaled = P / max_distance
```

This normalizes different hand sizes and camera distances.

---

## 4. Flattening

The normalized landmarks are flattened into a single feature vector.

```
21 landmarks
×

3 coordinates

=

63-dimensional feature vector
```

---

# Handling Missing Detections

Static image datasets occasionally contain:

- cropped hands
- finger occlusions
- poor lighting
- partially visible gestures

Some classes (such as **M**, **N**, and **T**) may therefore fail landmark detection.

Rather than terminating execution, the pipeline automatically:

- logs the failed image
- skips unusable samples
- continues dataset ingestion safely

---

# Neural Network Architecture

```
Input
63 Features
      │
      ▼
Dense (256)
ReLU
      │
Batch Normalization
      │
Dropout (40%)
      │
      ▼
Dense (128)
ReLU
      │
Batch Normalization
      │
Dropout (30%)
      │
      ▼
Dense (64)
ReLU
      │
Dropout (20%)
      │
      ▼
Dense (36)
Softmax
```

### Model Summary

- Input Features: **63**
- Hidden Layers: **256 → 128 → 64**
- Output Classes: **36**
- Activation: **ReLU**
- Output Activation: **Softmax**
- Optimizer: **Adam**
- Loss Function: **Categorical Cross-Entropy**

---

# Training Strategy

Training uses two callback mechanisms to maximize generalization.

### EarlyStopping

Stops training when validation performance no longer improves.

### ReduceLROnPlateau

Automatically lowers the learning rate after validation stagnation.

Together these techniques reduce overfitting while minimizing unnecessary training time.

---

# Usage

## Step 1 — Dataset Ingestion

Extract landmarks and populate the SQLite database.

```python
# Set MAX_IMAGES_PER_CLASS = 0 to process the complete dataset

ingest_dataset()
```

---

## Step 2 — Verify Dataset Balance

Inspect the number of samples per class.

```python
print(pivot.to_string())
```

---

## Step 3 — Train the Model

```python
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60
)
```

---

## Step 4 — Export TensorFlow Lite

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

---

## Step 5 — Real-Time Inference

Run webcam recognition.

```bash
python realtime_demo.py
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe Hands
- OpenCV
- SQLite
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Future Improvements

- Dynamic gesture recognition (ASL words and sentences)
- LSTM/Transformer temporal modeling
- Multi-hand recognition
- Model quantization
- ONNX export
- Android deployment
- Raspberry Pi optimization
- GPU acceleration
- Confidence threshold calibration

---

# Acknowledgements

This project utilizes:

- Google MediaPipe Hands
- TensorFlow/Keras
- OpenCV
- NumPy
- Scikit-learn

---

# License

This project is released under the **MIT License**.

---

# Author

**Jesse Jude**

Computer Science Student | Machine Learning & Software Engineering

GitHub: https://github.com/Jesse-jude
