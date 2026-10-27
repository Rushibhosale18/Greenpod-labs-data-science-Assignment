# 🥑 Avocado Ripening Stage Detection & Classification
### End-to-End Computer Vision Pipeline | Multi-Model Benchmarking | Edge Deployment Analysis
**GreenPod Labs — Data Science & Computer Vision Assignment 2**

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github)](https://github.com/Rushibhosale18/Greenpod-labs-data-science-Assignment)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B.svg?logo=streamlit)](https://share.streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikitlearn)](https://scikit-learn.org/)

---

## 📌 1. Project Overview

Post-harvest quality grading and shelf-life extension of fresh produce depend fundamentally on automated, non-destructive ripening stage classification. Avocados are climacteric fruits that undergo pronounced physiological transitions post-harvest: firmness decay, internal ethylene surge, chlorophyll degradation, and anthocyanin accumulation.

This repository implements, benchmarks, and provides an end-to-end computer vision workflow comparing two distinct paradigms across 4 ripening stages:
1. **Underripe / Hard:** Dark/bright green skin, high firmness, early harvest stage, non-edible.
2. **Breaking / Firm:** Transitioning skin tone, beginning of ethylene release and softening.
3. **Ripe / Soft:** Dark purple/black pebbled skin, optimal eating firmness and aroma.
4. **Overripe / Decay:** Senescent, sunken dark blemishes, fungal decay presence.

---

## 🏗️ 2. Project Architecture & Repository Structure

```
avocado_ripening_classification/
├── .gitignore                    # Git exclusions (data archives, model weights, cache)
├── README.md                     # Comprehensive setup guide, architecture, and results
├── requirements.txt              # Production and development dependencies
├── train_all_models.py           # Master benchmarking script for training both approaches
├── main_demo.py                  # Single-run inference script for automated demonstration
├── app.py                        # Interactive Streamlit Web Application & Sorting Dashboard
├── generate_pdf_report.py        # PDF generation script producing executive report
├── generate_docx_report.py       # DOCX report generation script
├── data/
│   ├── raw/                      # Downloaded and categorized dataset (1,090 images)
│   │   ├── underripe/            # 369 images
│   │   ├── breaking/             # 314 images
│   │   ├── ripe/                 # 201 images
│   │   └── overripe/             # 206 images
│   └── processed/                # Stratified Train/Val/Test CSV splits and cached features
│       ├── train_split.csv       # 763 images (70%)
│       ├── val_split.csv         # 163 images (15%)
│       ├── test_split.csv        # 164 images (15%)
│       └── split_summary.json    # Complete distribution metadata
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb       # Exploratory data analysis, color shift & glare audit
│   ├── 02_approach_1_classical_ml.ipynb     # CIELAB + HSV + LBP feature extraction + SVM/RF
│   ├── 03_approach_2_deep_learning.ipynb    # MobileNetV2 transfer learning, augmentations & loss
│   └── 04_model_evaluation_and_demo.ipynb   # Comparative evaluation, confusion matrix & demo
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Stratified splitting, PyTorch Dataset, and augmentations
│   ├── classical_model.py        # CIELAB/LBP feature engineering + SVM/RandomForest pipeline
│   └── dl_model.py               # MobileNetV2 transfer learning architecture & trainer
├── models/
│   ├── svm_model.joblib          # Trained Classical SVM pipeline (scaler + model)
│   ├── rf_model.joblib           # Trained Classical Random Forest pipeline
│   ├── mobilenet_v2_best.pth     # Best MobileNetV2 PyTorch weights
│   └── benchmark_metrics.json    # Serialized test set evaluation metrics
└── report/
    ├── Assignment2_Avocado_Report.pdf        # Publication-ready technical report
    ├── confusion_matrix_comparison.png       # Side-by-side test confusion matrices
    ├── model_performance_barchart.png        # Comparative performance metrics
    └── dl_learning_curves.png                # Loss and accuracy progression curves
```

---

## ⚡ 3. Quickstart & Installation Guide

### Prerequisites
- Python 3.8+ (Tested on Python 3.11)
- CPU or NVIDIA CUDA GPU

### Setup Environment
```bash
# 1. Clone repository
git clone https://github.com/Rushibhosale18/Greenpod-labs-data-science-Assignment.git
cd Greenpod-labs-data-science-Assignment

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🌐 4. Interactive Web Application & Cloud Deployment

The repository includes a production-ready interactive web application (`app.py`) built with Streamlit. It features:
- Dual-model real-time inference (Classical SVM vs MobileNetV2)
- Post-harvest shelf-life guidance and cold-chain sorting actions
- Interactive EDA with class distributions, color space breakdowns, and lighting glare audits
- Interactive model benchmarking matrices and confusion matrices
- Built-in PDF and DOCX technical report downloaders

### Run Locally:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 🚀 Option A: 1-Click Public Cloud Deployment (Streamlit Community Cloud)
The repository is completely configured with `packages.txt` (headless Linux OpenGL libraries) and serialized model weights for direct 1-click hosting:
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
2. Click **"New app"**.
3. Fill in:
   - **Repository:** `Rushibhosale18/Greenpod-labs-data-science-Assignment`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**
The application will build in ~2 minutes and provide a permanent public HTTPS URL (e.g., `https://rushibhosale18-greenpod-labs-avocado.streamlit.app`).

### 🌐 Option B: Deploy on Render (Render.com)
The repository includes native `render.yaml`, `Procfile`, `.python-version`, and `Dockerfile` for seamless deployment on Render:

**Method 1: Render Blueprints (Automatic Infrastructure-as-Code):**
1. Sign in to **[dashboard.render.com](https://dashboard.render.com)** with your GitHub account.
2. Click **"New +"** -> **"Blueprint"**.
3. Select `Rushibhosale18/Greenpod-labs-data-science-Assignment`.
4. Render automatically reads `render.yaml` and provisions your web service.
5. Click **"Apply"**.

**Method 2: Manual Web Service Setup on Render:**
1. Click **"New +"** -> **"Web Service"**.
2. Select repository: `Rushibhosale18/Greenpod-labs-data-science-Assignment`.
3. Configure the settings:
   - **Name:** `greenpod-avocado-ripening`
   - **Language:** `Python 3`
   - **Branch:** `main`
   - **Region:** `Oregon (US West)` or closest
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Instance Type:** `Free`
4. Click **"Create Web Service"**. Render will deploy your app at:
   `https://greenpod-avocado-ripening.onrender.com`

---

## 🚀 5. Execution & Usage

### 4.1 Run Single-Image Inference Demo (`main_demo.py`)
Demonstrate classification on an avocado image with side-by-side output from both Classical ML and Deep Learning:
```bash
# Run demonstration across all 4 ripening stages
python main_demo.py --demo_all

# Run on a specific image
python main_demo.py --image "data/raw/underripe/underripe_0001.jpg"
```

**Example Console Output:**
```
===========================================================================
AVOCADO RIPENING INFERENCE DEMO
Target Image: data/raw/underripe/underripe_0001.jpg (800x800 px)
Ground Truth Label: UNDERRIPE
---------------------------------------------------------------------------
Approach 1 (Classical ML - CIELAB+HSV+LBP): UNDERRIPE (Confidence: 99.6%)
  Latency: 18.2 ms
Approach 2 (MobileNetV2 Transfer Learning) : UNDERRIPE (Confidence: 99.9%)
  Latency: 12.8 ms
Model Agreement Status : AGREED
Ripening Stage Profile : Stage 1: Hard & Green. High firmness, early harvest stage, non-edible.
Automated Sorting Rec  : STORAGE / COLD TRANSIT: Hold in controlled atmosphere (low ethylene) for long-haul shipping.
===========================================================================
```

### 4.2 Re-Train & Benchmark All Models
To train the pipelines from scratch, extract feature vectors, compute test metrics, and generate comparison charts:
```bash
python train_all_models.py
```

### 4.3 Generate Technical PDF Report
To compile the formal submission report:
```bash
python generate_pdf_report.py
```

---

## 📊 5. Methodology & Technical Implementation

### Approach 1: Classical Computer Vision & ML
- **CIELAB Color Space:** Calculates 16-bin normalized histograms and statistical moments for $L^*$ (lightness), $a^*$ (green to red/purple chromatic axis), and $b^*$ (blue to yellow). The $a^*$ channel explicitly models chlorophyll breakdown into dark anthocyanins.
- **HSV Color Histograms:** Captures Hue and Saturation distributions invariant to illumination changes.
- **Local Binary Patterns (LBP):** Computes micro-texture representations capturing surface roughness differences between smooth unripened skin and bumpy/pebbled Hass avocado skin.
- **Classifier:** Support Vector Machine (RBF kernel, $C=10.0$, balanced class weighting) and Random Forest (150 estimators).

### Approach 2: Deep Learning Transfer Learning (MobileNetV2)
- **Backbone Architecture:** MobileNetV2 pretrained on ImageNet-1K. Features depthwise separable convolutions and inverted residuals with linear bottlenecks (~3.5M parameters).
- **Classification Head:** Linear(1280, 128) $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(128, 4).
- **Data Augmentation:** Random horizontal/vertical flips, mild rotations ($\pm 15^\circ$), and color jittering (brightness=0.15, contrast=0.15, saturation=0.15) to simulate optical glare and conveyor orientation shifts.
- **Loss Function:** Cross-Entropy Loss with inverse-frequency class weights to neutralize dataset imbalance.

---

## 📈 6. Metric Comparison Table (Held-Out Test Set: 164 Images)

| Evaluated Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approach 1: Classical SVM (RBF)** | **89.02%** | **88.51%** | **88.94%** | **88.62%** | **89.01%** | ~18 ms |
| **Approach 1: Random Forest** | **87.80%** | **87.41%** | **87.65%** | **87.48%** | **87.79%** | ~14 ms |
| **Approach 2: MobileNetV2 (Transfer Learning)** | **94.51%** | **94.20%** | **94.15%** | **94.12%** | **94.49%** | **~12 ms (GPU) / ~40 ms (CPU)** |

### Metric Rationale:
- **Macro F1-Score** was selected as the primary decisive metric because it weights each ripening stage equally, ensuring performance on the minority Ripe (18.4%) and Overripe (18.9%) classes is evaluated rigorously.
- **MobileNetV2** demonstrates superior discrimination, especially on the ambiguous boundary between late Breaking and early Ripe fruits.

---

## 📝 7. Mandatory End-of-Assignment Questions & Answers

### Question 1: In production, how does your model perform?
- **Inference Speed:** On an edge device (e.g., NVIDIA Jetson Orin Nano or Intel industrial CPU), MobileNetV2 processes frames in **12–25 ms on GPU** and **40–55 ms on CPU**. This translates to 20–30 frames per second, exceeding standard commercial sorting conveyor belt speeds (10–15 fruits/second).
- **Environmental & Lighting Shifts:** The model is robust under controlled industrial packing line illumination (diffuse CRI > 90). However, direct specular reflections from unshielded spotlights create localized white glare, which can bias RGB values. Our data augmentation pipeline (ColorJitter + contrast shifts) explicitly hardens the model against these artifacts.

### Question 2: What are the drawbacks of your model?
- **Color-Only Ambiguity:** Exterior skin color does not always correlate 100% with internal mesocarp softness. Certain avocado cultivars (e.g., Fuerte, Zutano) remain bright green even when fully ripe.
- **Single-Surface Camera View:** An overhead 2D camera inspects only ~40% of the avocado skin. Localized stem-end rot or underside bruising can escape detection without rotating rollers.
- **Continuous Ripening Boundary Noise:** Ripening is a continuous biological process; the boundary between late *breaking* and early *ripe* introduces ~5–8% subjective label noise among human graders.

### Question 3: If given more time, what would you improve?
- **Multi-Modal Sensing:** Combine RGB computer vision with non-destructive Near-Infrared (NIR) point spectroscopy (800–1100 nm) to estimate internal dry matter and oil content directly.
- **360-Degree Roller System:** Implement a dual-camera synchronized setup over biconical rotating rollers to achieve 100% surface inspection.
- **Edge INT8 Quantization:** Export PyTorch weights to ONNX Runtime and TensorRT INT8 for sub-5 ms latency on low-power edge accelerators.
- **Domain-Specific Self-Supervised Pretraining:** Pretrain the vision backbone on produce datasets using SimCLR or DINOv2 before fine-tuning.

### Question 4: Where did you spend the most time while solving your assignment?
- **40% Feature Engineering & Color Space Tuning:** Experimenting with RGB, HSV, and CIELAB color spaces to isolate the $a^*$ axis (chlorophyll breakdown) and designing multi-scale LBP texture descriptors.
- **30% Dataset Auditing & Handling Class Overlap:** Structuring the 1,090-image dataset, configuring stratified train/val/test splits, and setting up class-weighted loss functions to resolve the 63% vs 37% imbalance.
- **30% Model Benchmarking & Pipeline Architecture:** Implementing clean, reproducible OOP pipelines for classical ML and PyTorch transfer learning, running validation tracking, and compiling comparative reports.

---

## 👥 Author & Submission Details
- **Assignment:** GreenPod Labs Assignment 2 — Avocado Ripening Stage Detection & Classification
- **Repository Structure:** Standard Modular Python Package + Jupyter Notebooks + PDF Report
