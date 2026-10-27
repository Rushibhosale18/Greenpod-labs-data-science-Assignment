"""
Approach 1: Classical Computer Vision Feature Extraction + Classifier
Extracts CIELAB (a* green-to-red axis, L* lightness, b*), HSV color histograms,
and Local Binary Pattern (LBP) texture descriptors.
Trains and benchmarks Support Vector Machine (SVM) and Random Forest classifiers.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
import joblib

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
except ImportError:
    local_binary_pattern = None
    graycomatrix = None
    graycoprops = None

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS


def compute_local_binary_pattern(gray: np.ndarray, num_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Computes Local Binary Pattern (LBP) texture descriptor using vectorized NumPy.
    Captures micro-texture variations (smooth underripe skin vs pebbled/bumpy ripe skin).
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros(16, dtype=np.float32)

    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    # 8 circular/grid offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = gray[1+dy:h-1+dy, 1+dx:w-1+dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << i)

    # 16-bin normalized histogram
    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
    return hist


def extract_single_image_features(image_path: str, img_size: int = 128) -> np.ndarray:
    """
    Extracts comprehensive color and texture features from an avocado image:
    1. CIELAB color space:
       - a* channel: explicitly quantifies the green to purple/dark-red transition
       - L* channel: measures skin luminance and darkening
       - b* channel: measures yellow-blue nuances
       - 16-bin normalized histograms for L*, a*, b*
       - Channel statistical moments: mean, std, skewness
    2. HSV color space:
       - H (Hue) histogram (16 bins)
       - S (Saturation) histogram (8 bins)
    3. Texture Analysis via Local Binary Pattern (LBP):
       - Multi-scale LBP texture histograms capturing pebbled vs smooth skins
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for feature extraction.")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Resize to standardized dimension for consistent feature computation
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    features = []

    # --- 1. CIELAB Color Space Features ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 16-bin normalized histograms for each LAB channel
    hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256])
    hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256])
    hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256])

    hist_l = hist_l.flatten() / (hist_l.sum() + 1e-7)
    hist_a = hist_a.flatten() / (hist_a.sum() + 1e-7)
    hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)

    # Statistical moments
    lab_stats = [
        np.mean(l_chan), np.std(l_chan),
        np.mean(a_chan), np.std(a_chan),
        np.mean(b_chan), np.std(b_chan)
    ]
    features.extend(hist_l)
    features.extend(hist_a)
    features.extend(hist_b)
    features.extend(lab_stats)

    # --- 2. HSV Color Space Features ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])

    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
    features.extend(hist_h)
    features.extend(hist_s)

    # --- 3. Texture Descriptors (LBP) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_hist1 = compute_local_binary_pattern(gray, num_points=8, radius=1)
    features.extend(lbp_hist1)

    # Also extract gradient magnitude statistics
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([float(np.mean(grad_mag)), float(np.std(grad_mag))])

    return np.array(features, dtype=np.float32)


def extract_dataset_features(df: pd.DataFrame, desc: str = "Extracting Features") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts features for all image paths specified in a dataframe."""
    X_list = []
    y_list = []
    valid_paths = []

    print(f"[{desc}] Processing {len(df)} samples...")
    start_t = time.time()
    for idx, row in df.iterrows():
        try:
            feat = extract_single_image_features(row['image_path'])
            X_list.append(feat)
            y_list.append(int(row['class_idx']))
            valid_paths.append(row['image_path'])
        except Exception as e:
            print(f"Warning: Failed to extract features for {row['image_path']}: {e}")

    elapsed = time.time() - start_t
    print(f"[{desc}] Extracted {len(X_list)} feature vectors in {elapsed:.2f}s ({elapsed/len(X_list)*1000:.1f} ms/img)")
    return np.array(X_list), np.array(y_list), valid_paths


class ClassicalRipeningPipeline:
    """End-to-end Classical ML Pipeline with Feature Normalization and Classifier."""
    def __init__(self, model_type: str = 'svm', random_state: int = 42):
        self.model_type = model_type.lower()
        self.scaler = StandardScaler()
        self.random_state = random_state

        if self.model_type == 'svm':
            self.model = SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=random_state
            )
        elif self.model_type in ['rf', 'random_forest']:
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=14,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'svm' or 'rf'.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits scaler and classifier on training set."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class indices."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Runs full inference from a single raw image file."""
        t0 = time.time()
        features = extract_single_image_features(image_path).reshape(1, -1)
        pred_idx = int(self.predict(features)[0])
        probs = self.predict_proba(features)[0]
        inference_time_ms = (time.time() - t0) * 1000

        return {
            'predicted_class': IDX_TO_CLASS[pred_idx],
            'predicted_idx': pred_idx,
            'confidence': float(probs[pred_idx]),
            'all_probabilities': {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
            'inference_time_ms': inference_time_ms
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        """Calculates comprehensive classification metrics."""
        t0 = time.time()
        preds = self.predict(X)
        eval_time = time.time() - t0

        acc = accuracy_score(y, preds)
        prec_macro = precision_score(y, preds, average='macro', zero_division=0)
        rec_macro = recall_score(y, preds, average='macro', zero_division=0)
        f1_macro = f1_score(y, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y, preds)

        per_class_f1 = f1_score(y, preds, average=None, zero_division=0)
        per_class_rec = recall_score(y, preds, average=None, zero_division=0)
        per_class_prec = precision_score(y, preds, average=None, zero_division=0)

        results = {
            'split': split_name,
            'model_type': self.model_type,
            'accuracy': float(acc),
            'precision_macro': float(prec_macro),
            'recall_macro': float(rec_macro),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
            'per_class': {
                CLASS_NAMES[i]: {
                    'precision': float(per_class_prec[i]),
                    'recall': float(per_class_rec[i]),
                    'f1': float(per_class_f1[i])
                }
                for i in range(len(CLASS_NAMES))
            },
            'avg_inference_latency_ms': float((eval_time / len(X)) * 1000)
        }
        return results

    def save(self, filepath: str):
        """Saves trained pipeline (scaler + model) to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'model_type': self.model_type, 'scaler': self.scaler, 'model': self.model}, filepath)
        print(f"Classical model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads trained pipeline from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.scaler = data['scaler']
        instance.model = data['model']
        return instance
