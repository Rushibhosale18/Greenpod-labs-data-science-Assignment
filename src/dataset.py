"""
Avocado Ripening Stage Dataset Pipeline
Handles dataset discovery, stratified train/val/test splitting (70/15/15),
PyTorch Dataset wrappers, data augmentation transforms, and DataLoader creation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None

CLASS_NAMES = ['underripe', 'breaking', 'ripe', 'overripe']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}

STAGE_DESCRIPTIONS = {
    'underripe': 'Stage 1: Hard & Green. High firmness, early harvest stage, non-edible.',
    'breaking': 'Stage 2: Firm & Breaking. Slight color transition, beginning of ethylene release.',
    'ripe': 'Stage 3: Soft & Dark Purple/Black. Optimal eating quality, ripe, aromatic.',
    'overripe': 'Stage 4: Decay / Over-soft. Dark sunken spots, fungal presence, senescent.'
}


def scan_raw_dataset(raw_dir: str) -> pd.DataFrame:
    """
    Scans the raw dataset directory for images categorized by class.
    Expected structure:
        raw_dir/
            underripe/
            breaking/
            ripe/
            overripe/
    """
    records = []
    raw_path = Path(raw_dir)
    
    for cls_name in CLASS_NAMES:
        cls_folder = raw_path / cls_name
        if not cls_folder.exists():
            continue
        for file_path in cls_folder.iterdir():
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                records.append({
                    'image_path': str(file_path.resolve()),
                    'filename': file_path.name,
                    'class_name': cls_name,
                    'class_idx': CLASS_TO_IDX[cls_name]
                })
                
    df = pd.DataFrame(records)
    return df


def create_stratified_splits(
    raw_dir: str,
    processed_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    force_recreate: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates or loads stratified Train (70%), Validation (15%), and Test (15%) splits.
    Saves split metadata to processed_dir for reproducibility.
    """
    os.makedirs(processed_dir, exist_ok=True)
    train_csv = os.path.join(processed_dir, 'train_split.csv')
    val_csv = os.path.join(processed_dir, 'val_split.csv')
    test_csv = os.path.join(processed_dir, 'test_split.csv')
    meta_json = os.path.join(processed_dir, 'split_summary.json')

    if not force_recreate and os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        return train_df, val_df, test_df

    df = scan_raw_dataset(raw_dir)
    if len(df) == 0:
        raise ValueError(f"No images found in {raw_dir} across classes: {CLASS_NAMES}")

    # Stratified split: Train (70%) vs Temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df['class_idx'],
        random_state=random_state
    )

    # Split Temp (30%) equally into Val (15%) and Test (15%)
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_df['class_idx'],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    summary = {
        'total_images': len(df),
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df),
        'class_distribution': {
            'overall': df['class_name'].value_counts().to_dict(),
            'train': train_df['class_name'].value_counts().to_dict(),
            'val': val_df['class_name'].value_counts().to_dict(),
            'test': test_df['class_name'].value_counts().to_dict(),
        }
    }

    with open(meta_json, 'w') as f:
        json.dump(summary, f, indent=4)

    return train_df, val_df, test_df


def get_transforms(img_size: int = 224):
    """
    Returns data augmentation and normalization transforms.
    Augmentation strategies include:
      - Random horizontal flip (simulates conveyor orientation shifts)
      - Mild rotation (-15 to +15 deg)
      - Color jitter (mild brightness/contrast variance to handle lighting glare)
      - Normalization via ImageNet mean & std
    """
    if transforms is None:
        return None, None

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_test_transform


class AvocadoPyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for Avocado images."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = int(row['class_idx'])

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_path


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """Builds and returns DataLoaders for train, val, and test splits."""
    train_tf, val_tf = get_transforms(img_size=img_size)

    train_ds = AvocadoPyTorchDataset(train_df, transform=train_tf)
    val_ds = AvocadoPyTorchDataset(val_df, transform=val_tf)
    test_ds = AvocadoPyTorchDataset(test_df, transform=val_tf)

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return loaders


if __name__ == '__main__':
    raw_dir = r"C:\Users\Rushi\.gemini\antigravity\scratch\avocado_ripening_classification\data\raw"
    proc_dir = r"C:\Users\Rushi\.gemini\antigravity\scratch\avocado_ripening_classification\data\processed"
    tr, va, te = create_stratified_splits(raw_dir, proc_dir)
    print(f"Stratified split summary: Train={len(tr)}, Val={len(va)}, Test={len(te)}")
