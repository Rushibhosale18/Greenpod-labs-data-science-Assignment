"""
Approach 2: Deep Learning Transfer Learning Architecture (MobileNetV2 / ResNet18)
Implements edge-optimized transfer learning with pre-trained ImageNet weights,
custom classification head, early stopping, and comprehensive evaluation metrics.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ImportError:
    torch = None
    nn = object
    models = None
    transforms = None

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from src.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    if torch is not None and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_avocado_classifier(
    architecture: str = 'mobilenet_v2',
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Constructs an edge-optimized transfer learning model.
    Options:
      - 'mobilenet_v2': ~3.5M parameters, inverted residual blocks, ideal for edge/conveyor belts.
      - 'resnet18': ~11.7M parameters, residual skip connections.
    """
    if models is None:
        raise ImportError("torchvision.models is required to construct the classifier.")

    arch = architecture.lower()
    weights = 'DEFAULT' if pretrained else None

    if arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    elif arch == 'resnet18':
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose 'mobilenet_v2' or 'resnet18'.")

    return model


class DeepLearningRipeningPipeline:
    """Deep Learning Training, Inference, and Evaluation Manager."""
    def __init__(
        self,
        architecture: str = 'mobilenet_v2',
        num_classes: int = 4,
        device: Optional[torch.device] = None,
        freeze_backbone: bool = True
    ):
        self.architecture = architecture
        self.num_classes = num_classes
        self.device = device if device is not None else get_device()
        self.model = build_avocado_classifier(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=freeze_backbone
        ).to(self.device)

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def train_pipeline(
        self,
        train_loader,
        val_loader,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Executes model training with validation monitoring and early stopping.
        Saves best checkpoint based on validation macro F1-score.
        """
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1_macro': []
        }

        best_val_f1 = -1.0

        print(f"\n--- Starting Deep Learning Training ({self.architecture} on {self.device}) ---")
        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels, _ in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total

            # Validation
            val_metrics = self.evaluate(val_loader, split_name="Validation")
            scheduler.step(val_metrics['f1_macro'])

            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1_macro'].append(val_metrics['f1_macro'])

            elapsed = time.time() - t_epoch_start
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.3f}, "
                f"Val F1: {val_metrics['f1_macro']:.3f}"
            )

            # Checkpoint best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                if checkpoint_path:
                    self.save(checkpoint_path)
                    print(f"  --> Saved new best checkpoint with Val F1: {best_val_f1:.4f}")

        return history

    def evaluate(self, loader, split_name: str = "Test") -> Dict[str, Any]:
        """Runs evaluation over a DataLoader and computes full metric breakdown."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        total_samples = 0
        t0 = time.time()

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                total_samples += labels.size(0)

        elapsed = time.time() - t0
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        avg_loss = total_loss / max(total_samples, 1)
        acc = accuracy_score(all_targets, all_preds)
        prec_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        rec_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0)
        per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0)

        results = {
            'split': split_name,
            'architecture': self.architecture,
            'loss': float(avg_loss),
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
            'avg_inference_latency_ms': float((elapsed / max(total_samples, 1)) * 1000)
        }
        return results

    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Performs single-image inference."""
        self.model.eval()
        t0 = time.time()

        img = Image.open(image_path).convert('RGB')
        tensor = self.eval_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        inference_time_ms = (time.time() - t0) * 1000

        return {
            'predicted_class': IDX_TO_CLASS[pred_idx],
            'predicted_idx': pred_idx,
            'confidence': float(probs[pred_idx]),
            'all_probabilities': {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            'inference_time_ms': inference_time_ms
        }

    def save(self, filepath: str):
        """Saves model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'architecture': self.architecture,
            'state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None):
        """Loads model weights."""
        try:
            checkpoint = torch.load(filepath, map_location=device if device else 'cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device if device else 'cpu')

        instance = cls(
            architecture=checkpoint.get('architecture', 'mobilenet_v2'),
            num_classes=checkpoint.get('num_classes', 4),
            device=device
        )
        instance.model.load_state_dict(checkpoint['state_dict'])
        instance.model.eval()
        return instance
