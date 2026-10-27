"""
Avocado Ripening Stage Classification - Single-Run Demo Script
Demonstrates inference on input avocado images using both:
  - Approach 1: Classical CV (CIELAB + HSV + LBP) + SVM Classifier
  - Approach 2: Deep Learning (MobileNetV2 Transfer Learning)
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

from src.dataset import CLASS_NAMES, STAGE_DESCRIPTIONS, IDX_TO_CLASS
from src.classical_model import ClassicalRipeningPipeline, extract_single_image_features
from src.dl_model import DeepLearningRipeningPipeline, get_device

SORTING_ACTIONS = {
    'underripe': 'STORAGE / COLD TRANSIT: Hold in controlled atmosphere (low ethylene) for long-haul shipping.',
    'breaking': 'DISTRIBUTION CENTER: Move to regional distribution; stage for ripening room initiation.',
    'ripe': 'RETAIL READY: Route immediately to supermarket display / consumer sales (peak firmness & flavor).',
    'overripe': 'REJECT / COMPOST / PROCESSING: Remove from fresh produce sorting line; divert to puree or discard.'
}


def extract_avocado_features(image_path: str):
    """Reference wrapper matching assignment specification Section 5.2."""
    return extract_single_image_features(image_path)


def run_avocado_demo(image_path: str, classical_model, dl_model, class_names=CLASS_NAMES, true_stage: str = None):
    """
    Executes and compares Approach 1 (Classical ML) and Approach 2 (MobileNetV2)
    on a single avocado image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 1. Load image via OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not decode image at {image_path}")
    h, w, c = img.shape

    # 2. Predict using Classical ML Approach
    t0_classic = time.time()
    features = extract_avocado_features(image_path).reshape(1, -1)
    classical_pred_idx = classical_model.predict(features)[0]
    classical_pred = class_names[classical_pred_idx] if isinstance(classical_pred_idx, (int, np.integer)) else classical_pred_idx
    classical_probs = classical_model.predict_proba(features)[0] if hasattr(classical_model, 'predict_proba') else None
    t_classic_ms = (time.time() - t0_classic) * 1000

    # 3. Predict using Deep Learning Approach (MobileNetV2)
    t0_dl = time.time()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0

    # Apply ImageNet normalization: (x - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_resized - mean) / std

    tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    device = next(dl_model.parameters()).device
    tensor = tensor.to(device)

    dl_model.eval()
    with torch.no_grad():
        outputs = dl_model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        dl_pred_idx = int(torch.argmax(outputs, dim=1).item())
        dl_pred = class_names[dl_pred_idx]
    t_dl_ms = (time.time() - t0_dl) * 1000

    # Print Formatted Results
    print("=" * 75)
    print(f"AVOCADO RIPENING INFERENCE DEMO")
    print(f"Target Image: {image_path} ({w}x{h} px)")
    if true_stage:
        print(f"Ground Truth Label: {true_stage.upper()}")
    print("-" * 75)

    classic_conf_str = f" (Confidence: {classical_probs[classical_pred_idx]*100:.1f}%)" if classical_probs is not None else ""
    print(f"Approach 1 (Classical ML - CIELAB+HSV+LBP): {classical_pred.upper()}{classic_conf_str}")
    print(f"  Latency: {t_classic_ms:.1f} ms")

    print(f"Approach 2 (MobileNetV2 Transfer Learning) : {dl_pred.upper()} (Confidence: {probs[dl_pred_idx]*100:.1f}%)")
    print(f"  Latency: {t_dl_ms:.1f} ms")

    consensus = "AGREED" if classical_pred == dl_pred else "DISCREPANCY (Edge boundary case)"
    print(f"Model Agreement Status : {consensus}")
    print(f"Ripening Stage Profile : {STAGE_DESCRIPTIONS.get(dl_pred, 'N/A')}")
    print(f"Automated Sorting Rec  : {SORTING_ACTIONS.get(dl_pred, 'N/A')}")
    print("=" * 75)

    return {
        'image_path': image_path,
        'classical_pred': classical_pred,
        'dl_pred': dl_pred,
        'dl_confidence': float(probs[dl_pred_idx]),
        'classical_latency_ms': t_classic_ms,
        'dl_latency_ms': t_dl_ms
    }


def main():
    parser = argparse.ArgumentParser(description="Avocado Ripening Stage Classification Demo")
    parser.add_argument('--image', type=str, default=None, help="Path to single image for inference.")
    parser.add_argument('--classical_model_path', type=str,
                        default=os.path.join(PROJECT_DIR, 'models', 'svm_model.joblib'),
                        help="Path to trained classical model (.joblib).")
    parser.add_argument('--dl_model_path', type=str,
                        default=os.path.join(PROJECT_DIR, 'models', 'mobilenet_v2_best.pth'),
                        help="Path to trained DL weights (.pth).")
    parser.add_argument('--demo_all', action='store_true',
                        help="Runs inference on one sample from each of the 4 ripening stages.")

    args = parser.parse_args()

    # Load Classical Pipeline
    if not os.path.exists(args.classical_model_path):
        print(f"Error: Classical model not found at {args.classical_model_path}.")
        print("Please run `python train_all_models.py` first to generate trained models.")
        return

    print("Loading models...")
    classical_pipeline = ClassicalRipeningPipeline.load(args.classical_model_path)

    # Load Deep Learning Pipeline
    if not os.path.exists(args.dl_model_path):
        print(f"Error: Deep learning model not found at {args.dl_model_path}.")
        print("Please run `python train_all_models.py` first to generate trained models.")
        return

    dl_pipeline = DeepLearningRipeningPipeline.load(args.dl_model_path)
    dl_model = dl_pipeline.model

    if args.image:
        run_avocado_demo(args.image, classical_pipeline, dl_model)
    else:
        # Default: demonstrate across all 4 stages using sample test images
        print("\nNo individual image specified. Executing demonstration across all 4 stages...")
        raw_dir = os.path.join(PROJECT_DIR, 'data', 'raw')
        for stage in CLASS_NAMES:
            stage_dir = os.path.join(raw_dir, stage)
            if os.path.exists(stage_dir):
                files = [f for f in os.listdir(stage_dir) if f.lower().endswith(('.jpg', '.png'))]
                if files:
                    sample_img = os.path.join(stage_dir, files[0])
                    run_avocado_demo(sample_img, classical_pipeline, dl_model, true_stage=stage)
                    print()


if __name__ == '__main__':
    main()
