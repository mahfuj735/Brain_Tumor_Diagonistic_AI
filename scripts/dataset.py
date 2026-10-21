"""
Mask-XAI-Net: Universal Dataset & Preprocessing Pipeline
=======================================================
Supports:
1. Cheng Figshare (Folds 1-3: Train, Fold 4: Val, Fold 5: Test | 3 classes: glioma, meningioma, pituitary)
2. BRISC 2025 (Classification 4 classes & Segmentation image-mask pairs)
3. Nickparvar MRI (4 classes: glioma, meningioma, notumor, pituitary)

Technical Specifications:
- Brain Contour Cropping with 5% safety margin
- CLAHE Contrast Enhancement (clipLimit=2.0, tileGridSize=(8, 8))
- Dual Interpolation (Image: Bilinear, Mask: Nearest Neighbor)
- Synchronized Albumentations Transforms (train split)
- ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Dynamic Zero-Mask synthesis for healthy brains and has_mask flag
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Universal Class Mappings
# All 3 tumor classes are strictly aligned across all benchmarks:
# 0: glioma, 1: meningioma, 2: pituitary, 3: no_tumor / notumor
CLASS_TO_IDX_4 = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2,
    'no_tumor': 3,
    'notumor': 3
}

CLASS_TO_IDX_3 = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2
}

IDX_TO_CLASS_4 = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}
IDX_TO_CLASS_3 = {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_brain_contour(image: np.ndarray, mask: np.ndarray = None, pad_ratio: float = 0.05):
    """
    Crops the MRI image to the skull/brain boundary to eliminate empty black background
    while maintaining a pad_ratio (default 5%) safety margin so peripheral meningiomas
    touching the skull boundary are not clipped.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Smooth image to remove high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to fill potential holes inside the brain
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = gray.shape
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # Ensure contour is meaningful (> 5% of total image area)
        if area > (h_img * w_img * 0.05):
            x, y, w, h = cv2.boundingRect(c)
            pad_w = int(w * pad_ratio)
            pad_h = int(h * pad_ratio)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)
            
            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2] if mask is not None else None
            return cropped_image, cropped_mask

    # Fallback to original if no valid contour
    return image, mask


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance
    contrast between tumor tissue and healthy parenchyma.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:
        return clahe.apply(image)
    elif len(image.shape) == 3:
        # Convert to LAB, apply CLAHE to L-channel, and convert back
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return image


def get_transforms(split: str = 'train', image_size: int = 224):
    """
    Returns coupled Albumentations transforms for train, val, or test splits.
    Train split includes synchronized horizontal flip and affine perturbations.
    """
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                p=0.5
            ),
        ])
    else:
        return None


class BaseBrainTumorDataset(Dataset):
    """
    Abstract base class providing standard preprocessing pipeline:
    1. Contour crop (5% margin)
    2. CLAHE enhancement
    3. Dual-interpolation resize (Bilinear for image, Nearest for mask)
    4. Albumentations transform (synchronized)
    5. Normalization & PyTorch Tensor conversion
    """
    def __init__(
        self, 
        split: str = 'train', 
        image_size: int = 224, 
        apply_crop: bool = True,
        apply_clahe_enhancement: bool = True,
        augment: bool = True
    ):
        self.split = split
        self.image_size = image_size
        self.apply_crop = apply_crop
        self.apply_clahe_enhancement = apply_clahe_enhancement
        self.transform = get_transforms(split, image_size) if augment and split == 'train' else None

    def preprocess_pair(self, image_np: np.ndarray, mask_np: np.ndarray = None):
        """
        Processes an image and optional mask through the standard pipeline.
        image_np: uint8 array (H, W) or (H, W, 3)
        mask_np: uint8 array (H, W) with {0, 255} or None
        """
        # Ensure image is in RGB
        if len(image_np.shape) == 2:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_np.copy()

        # Step 1: Brain Contour Crop
        if self.apply_crop:
            image_rgb, mask_np = crop_brain_contour(image_rgb, mask_np, pad_ratio=0.05)

        # Step 2: CLAHE Enhancement
        if self.apply_clahe_enhancement:
            image_rgb = apply_clahe(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8))

        # Step 3: Dual Interpolation Resizing
        image_resized = cv2.resize(
            image_rgb, 
            (self.image_size, self.image_size), 
            interpolation=cv2.INTER_LINEAR
        )
        
        if mask_np is not None:
            mask_resized = cv2.resize(
                mask_np, 
                (self.image_size, self.image_size), 
                interpolation=cv2.INTER_NEAREST
            )
            # Re-threshold to guarantee pure binary values {0.0, 1.0}
            mask_binary = (mask_resized > 127).astype(np.float32)
        else:
            mask_binary = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Step 4: Synchronized Augmentation (Albumentations)
        if self.transform is not None:
            transformed = self.transform(image=image_resized, mask=mask_binary)
            image_resized = transformed['image']
            mask_binary = transformed['mask']

        # Step 5: Normalization & Tensor Conversion
        # Normalize image to [0, 1] then ImageNet mean/std
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - IMAGENET_MEAN) / IMAGENET_STD
        
        # PyTorch format: (3, H, W)
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
        
        # Mask tensor format: (1, H, W)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return image_tensor, mask_tensor


class ChengFigshareDataset(BaseBrainTumorDataset):
    """
    Cheng Figshare Benchmark:
    - 3 Classes: glioma (0), meningioma (1), pituitary (2)
    - Ground-truth binary segmentation masks available for 100% of images
    - Splits: 'train' (Folds 1-3), 'val' (Fold 4), 'test' (Fold 5)
    """
    def __init__(
        self, 
        base_dir: str = r"datasets/cheng_figshare", 
        split: str = 'train', 
        image_size: int = 224, 
        apply_crop: bool = True,
        apply_clahe_enhancement: bool = True,
        augment: bool = True
    ):
        super().__init__(split, image_size, apply_crop, apply_clahe_enhancement, augment)
        self.base_dir = base_dir
        csv_path = os.path.join(base_dir, "cheng_figshare_metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cheng Figshare metadata not found at {csv_path}")

        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row['image_path'])
        mask_path = os.path.join(self.base_dir, row['mask_path'])

        # Load image (lossless PNG)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")

        # Class label
        cls_name = row['class_name']
        label = CLASS_TO_IDX_3[cls_name]

        image_tensor, mask_tensor = self.preprocess_pair(image, mask)
        has_mask = torch.tensor(True, dtype=torch.bool)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'label': label_tensor,
            'has_mask': has_mask,
            'patient_id': row['patient_id'],
            'slice_id': row['slice_id'],
            'dataset': 'cheng_figshare'
        }


class BRISC2025Dataset(BaseBrainTumorDataset):
    """
    BRISC 2025 Benchmark:
    - 4 Classes: glioma (0), meningioma (1), pituitary (2), no_tumor (3)
    - Tasks supported:
      * 'classification': All 4 classes (healthy slices have zero-mask synthesized)
      * 'segmentation': Tumor classes with image-mask pairs
    - Splits: 'train', 'val', 'test'
    """
    def __init__(
        self, 
        base_dir: str = r"datasets/brisc2025", 
        split: str = 'train', 
        task: str = 'classification',
        image_size: int = 224, 
        apply_crop: bool = True,
        apply_clahe_enhancement: bool = True,
        augment: bool = True
    ):
        super().__init__(split, image_size, apply_crop, apply_clahe_enhancement, augment)
        self.base_dir = base_dir
        self.task = task
        csv_path = os.path.join(base_dir, "manifest.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"BRISC 2025 manifest not found at {csv_path}")

        df = pd.read_csv(csv_path)
        # Filter by task and split
        seg_masks = df[(df['task'] == 'segmentation') & (df['is_mask'] == True)]
        self.mask_lookup = dict(zip(seg_masks['filename'], seg_masks['relative_path']))

        if task == 'classification':
            self.df = df[(df['task'] == 'classification') & (df['split'] == split)].reset_index(drop=True)
        elif task == 'segmentation':
            # In segmentation task, we only select image rows (is_mask == False)
            self.df = df[(df['task'] == 'segmentation') & (df['split'] == split) & (~df['is_mask'])].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'segmentation'.")

        if len(self.df) == 0:
            raise ValueError(f"No samples found for task '{task}', split '{split}' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row['relative_path']
        img_path = os.path.join(self.base_dir, img_rel_path)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        cls_name = row['tumor_label']
        label = CLASS_TO_IDX_4[cls_name]
        filename = row['filename']

        # Determine mask
        mask = None
        has_mask_val = False

        if filename in self.mask_lookup:
            mask_rel_path = self.mask_lookup[filename]
            mask_full_path = os.path.join(self.base_dir, mask_rel_path)
            if os.path.exists(mask_full_path):
                mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
                has_mask_val = True

        if cls_name == 'no_tumor':
            # For healthy brain, mask is definitively all zeros
            mask = np.zeros_like(image, dtype=np.uint8)
            has_mask_val = True

        image_tensor, mask_tensor = self.preprocess_pair(image, mask)
        has_mask = torch.tensor(has_mask_val, dtype=torch.bool)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'label': label_tensor,
            'has_mask': has_mask,
            'filename': filename,
            'dataset': 'brisc2025'
        }


class NickparvarDataset(BaseBrainTumorDataset):
    """
    Nickparvar MRI Benchmark:
    - 4 Classes: glioma (0), meningioma (1), pituitary (2), notumor (3)
    - Pure classification (No physical masks available on disk)
    - Returns zero-mask tensor with has_mask=False so loss function ignores segmentation loss
    - Splits: 'train', 'val', 'test'
    """
    def __init__(
        self, 
        base_dir: str = r"datasets/nickparvar_mri", 
        split: str = 'train', 
        image_size: int = 224, 
        apply_crop: bool = True,
        apply_clahe_enhancement: bool = True,
        augment: bool = True
    ):
        super().__init__(split, image_size, apply_crop, apply_clahe_enhancement, augment)
        self.base_dir = base_dir
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Nickparvar split directory not found: {split_dir}")

        self.samples = []
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        for c in classes:
            c_dir = os.path.join(split_dir, c)
            if os.path.exists(c_dir):
                for fname in sorted(os.listdir(c_dir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(c_dir, fname), CLASS_TO_IDX_4[c], c, fname))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{split}' in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cls_name, fname = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        # Check for merged pseudo mask
        base_name = os.path.splitext(fname)[0]
        mask_path = os.path.join(self.base_dir, "masks", self.split, cls_name, f"{base_name}_mask.png")
        mask = None
        has_mask_val = False

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            has_mask_val = True
        elif cls_name == 'notumor':
            # Healthy brain guaranteed all-zero mask
            mask = np.zeros_like(image, dtype=np.uint8)
            has_mask_val = True

        image_tensor, mask_tensor = self.preprocess_pair(image, mask_np=mask)
        has_mask = torch.tensor(has_mask_val, dtype=torch.bool)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'label': label_tensor,
            'has_mask': has_mask,
            'filename': fname,
            'dataset': 'nickparvar_mri'
        }


def get_dataset(
    dataset_name: str,
    split: str = 'train',
    image_size: int = 224,
    apply_crop: bool = True,
    apply_clahe_enhancement: bool = True,
    augment: bool = True,
    task: str = 'classification'
) -> Dataset:
    """
    Factory function to instantiate any of the 3 benchmark datasets.
    """
    name = dataset_name.lower().strip()
    if name in ['cheng_figshare', 'figshare', 'cheng']:
        return ChengFigshareDataset(
            split=split, 
            image_size=image_size, 
            apply_crop=apply_crop, 
            apply_clahe_enhancement=apply_clahe_enhancement, 
            augment=augment
        )
    elif name in ['brisc2025', 'brisc']:
        return BRISC2025Dataset(
            split=split, 
            task=task,
            image_size=image_size, 
            apply_crop=apply_crop, 
            apply_clahe_enhancement=apply_clahe_enhancement, 
            augment=augment
        )
    elif name in ['nickparvar_mri', 'nickparvar']:
        return NickparvarDataset(
            split=split, 
            image_size=image_size, 
            apply_crop=apply_crop, 
            apply_clahe_enhancement=apply_clahe_enhancement, 
            augment=augment
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Options: 'cheng_figshare', 'brisc2025', 'nickparvar_mri'")


def get_dataloader(
    dataset_name: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 0,
    image_size: int = 224,
    apply_crop: bool = True,
    apply_clahe_enhancement: bool = True,
    augment: bool = True,
    task: str = 'classification'
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader for the requested dataset and split.
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = get_dataset(
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
        apply_crop=apply_crop,
        apply_clahe_enhancement=apply_clahe_enhancement,
        augment=augment,
        task=task
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    print("Mask-XAI-Net Dataset Module Defined.")
