"""
Master Training and Benchmarking Script
Trains:
  1. Classical ML Pipeline (SVM & Random Forest) using CIELAB + HSV + LBP
  2. Deep Learning Transfer Learning (MobileNetV2)
Evaluates on the held-out test split (164 images) and exports metrics and comparison plots.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

from src.dataset import create_stratified_splits, get_dataloaders, CLASS_NAMES
from src.classical_model import (
    extract_dataset_features, ClassicalRipeningPipeline
)
from src.dl_model import DeepLearningRipeningPipeline, get_device
import torch


def run_training_pipeline():
    raw_dir = os.path.join(PROJECT_DIR, 'data', 'raw')
    proc_dir = os.path.join(PROJECT_DIR, 'data', 'processed')
    models_dir = os.path.join(PROJECT_DIR, 'models')
    reports_dir = os.path.join(PROJECT_DIR, 'report')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Stratified Data Splitting (70% Train, 15% Val, 15% Test)")
    print("=" * 70)
    train_df, val_df, test_df = create_stratified_splits(raw_dir, proc_dir, force_recreate=False)
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # =========================================================================
    # APPROACH 1: CLASSICAL CV + MACHINE LEARNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Approach 1 - Classical CV (CIELAB + HSV + LBP Feature Extraction)")
    print("=" * 70)

    features_cache = os.path.join(proc_dir, 'extracted_features.npz')
    if os.path.exists(features_cache):
        print(f"Loading cached feature vectors from {features_cache}...")
        cache_data = np.load(features_cache)
        X_train, y_train = cache_data['X_train'], cache_data['y_train']
        X_val, y_val = cache_data['X_val'], cache_data['y_val']
        X_test, y_test = cache_data['X_test'], cache_data['y_test']
    else:
        X_train, y_train, _ = extract_dataset_features(train_df, "Train Split")
        X_val, y_val, _ = extract_dataset_features(val_df, "Val Split")
        X_test, y_test, _ = extract_dataset_features(test_df, "Test Split")
        np.savez_compressed(
            features_cache,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        print(f"Saved extracted features cache to {features_cache}")

    print(f"Feature vector dimensionality: {X_train.shape[1]}")

    # Train SVM Classifier
    print("\nTraining Classical SVM (RBF Kernel, Balanced Weights)...")
    svm_pipeline = ClassicalRipeningPipeline(model_type='svm', random_state=42)
    t0 = time.time()
    svm_pipeline.fit(X_train, y_train)
    svm_train_time = time.time() - t0
    svm_test_metrics = svm_pipeline.evaluate(X_test, y_test, split_name="Test")
    svm_test_metrics['train_time_sec'] = svm_train_time
    svm_pipeline.save(os.path.join(models_dir, 'svm_model.joblib'))

    # Train Random Forest Classifier
    print("\nTraining Classical Random Forest (150 Trees, Balanced Weights)...")
    rf_pipeline = ClassicalRipeningPipeline(model_type='rf', random_state=42)
    t0 = time.time()
    rf_pipeline.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    rf_test_metrics = rf_pipeline.evaluate(X_test, y_test, split_name="Test")
    rf_test_metrics['train_time_sec'] = rf_train_time
    rf_pipeline.save(os.path.join(models_dir, 'rf_model.joblib'))

    print(f"SVM Test Accuracy: {svm_test_metrics['accuracy']:.4f} | Macro F1: {svm_test_metrics['f1_macro']:.4f}")
    print(f"RF  Test Accuracy: {rf_test_metrics['accuracy']:.4f} | Macro F1: {rf_test_metrics['f1_macro']:.4f}")

    # =========================================================================
    # APPROACH 2: DEEP LEARNING (TRANSFER LEARNING MOBILENETV2)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Approach 2 - Deep Learning Transfer Learning (MobileNetV2)")
    print("=" * 70)

    device = get_device()
    print(f"Executing Deep Learning on hardware: {device}")

    # Compute class weights for cross-entropy to mitigate imbalance
    class_counts = train_df['class_idx'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights_arr = total_samples / (len(CLASS_NAMES) * class_counts)
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)

    loaders = get_dataloaders(train_df, val_df, test_df, batch_size=32, img_size=224)

    dl_pipeline = DeepLearningRipeningPipeline(
        architecture='mobilenet_v2',
        num_classes=len(CLASS_NAMES),
        device=device,
        freeze_backbone=True
    )

    mobilenet_ckpt = os.path.join(models_dir, 'mobilenet_v2_best.pth')
    t0 = time.time()
    history = dl_pipeline.train_pipeline(
        loaders['train'],
        loaders['val'],
        epochs=12,
        lr=1e-3,
        class_weights=class_weights_tensor,
        checkpoint_path=mobilenet_ckpt
    )
    dl_train_time = time.time() - t0

    # Load best checkpoint for test evaluation
    best_dl = DeepLearningRipeningPipeline.load(mobilenet_ckpt, device=device)
    dl_test_metrics = best_dl.evaluate(loaders['test'], split_name="Test")
    dl_test_metrics['train_time_sec'] = dl_train_time

    print(f"\nMobileNetV2 Test Accuracy: {dl_test_metrics['accuracy']:.4f} | Macro F1: {dl_test_metrics['f1_macro']:.4f}")

    # =========================================================================
    # STEP 4: METRIC BENCHMARK COMPARISON & EXPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Compiling Metric Benchmarks & Plots")
    print("=" * 70)

    benchmark_summary = {
        'SVM (Classical)': {
            'Accuracy': svm_test_metrics['accuracy'],
            'Macro Precision': svm_test_metrics['precision_macro'],
            'Macro Recall': svm_test_metrics['recall_macro'],
            'Macro F1-Score': svm_test_metrics['f1_macro'],
            'Weighted F1-Score': svm_test_metrics['f1_weighted'],
            'Inference Latency (ms)': svm_test_metrics['avg_inference_latency_ms'],
            'Training Time (s)': svm_train_time
        },
        'Random Forest (Classical)': {
            'Accuracy': rf_test_metrics['accuracy'],
            'Macro Precision': rf_test_metrics['precision_macro'],
            'Macro Recall': rf_test_metrics['recall_macro'],
            'Macro F1-Score': rf_test_metrics['f1_macro'],
            'Weighted F1-Score': rf_test_metrics['f1_weighted'],
            'Inference Latency (ms)': rf_test_metrics['avg_inference_latency_ms'],
            'Training Time (s)': rf_train_time
        },
        'MobileNetV2 (Transfer Learning)': {
            'Accuracy': dl_test_metrics['accuracy'],
            'Macro Precision': dl_test_metrics['precision_macro'],
            'Macro Recall': dl_test_metrics['recall_macro'],
            'Macro F1-Score': dl_test_metrics['f1_macro'],
            'Weighted F1-Score': dl_test_metrics['f1_weighted'],
            'Inference Latency (ms)': dl_test_metrics['avg_inference_latency_ms'],
            'Training Time (s)': dl_train_time
        }
    }

    df_benchmark = pd.DataFrame(benchmark_summary).T
    print(df_benchmark.to_string())

    # Save benchmark table as JSON & CSV
    with open(os.path.join(models_dir, 'benchmark_metrics.json'), 'w') as f:
        json.dump(benchmark_summary, f, indent=4)
    df_benchmark.to_csv(os.path.join(models_dir, 'benchmark_metrics.csv'))

    # Plot Confusion Matrices Side by Side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cms = [
        (svm_test_metrics['confusion_matrix'], "Classical SVM (RBF)"),
        (rf_test_metrics['confusion_matrix'], "Classical Random Forest"),
        (dl_test_metrics['confusion_matrix'], "MobileNetV2 Transfer Learning")
    ]

    for ax, (cm_data, title) in zip(axes, cms):
        cm_norm = np.array(cm_data).astype('float') / (np.array(cm_data).sum(axis=1)[:, np.newaxis] + 1e-7)
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=False
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Stage')
        ax.set_ylabel('True Stage')

    plt.suptitle('Confusion Matrix Comparison on Test Set (164 images)', fontsize=14, y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, 'confusion_matrix_comparison.png')
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix comparison plot to {cm_path}")

    # Plot Metrics Bar Chart
    metrics_to_plot = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
    ax = df_benchmark[metrics_to_plot].plot(kind='bar', figsize=(11, 5), colormap='viridis', rot=0)
    plt.title('Avocado Ripening Classification - Model Performance Comparison', fontsize=13, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    bar_path = os.path.join(reports_dir, 'model_performance_barchart.png')
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved performance bar chart to {bar_path}")

    # Plot Training Loss & Accuracy Curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss', color='darkorange')
    ax1.plot(epochs_range, history['val_loss'], 's--', label='Val Loss', color='dodgerblue')
    ax1.set_title('MobileNetV2: Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(epochs_range, history['train_acc'], 'o-', label='Train Accuracy', color='darkgreen')
    ax2.plot(epochs_range, history['val_acc'], 's--', label='Val Accuracy', color='purple')
    ax2.set_title('MobileNetV2: Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    curve_path = os.path.join(reports_dir, 'dl_learning_curves.png')
    plt.savefig(curve_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved deep learning training curves to {curve_path}")

    print("\nAll models trained, evaluated, and saved successfully!")
    return benchmark_summary


if __name__ == '__main__':
    run_training_pipeline()
