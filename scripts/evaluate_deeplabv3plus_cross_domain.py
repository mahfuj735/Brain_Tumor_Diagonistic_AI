import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, f1_score
from concurrent.futures import ThreadPoolExecutor

# Use all CPU cores for inference
torch.set_num_threads(os.cpu_count())

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE    = 256
BATCH_SIZE    = 32

def crop_brain_contour(image: np.ndarray, mask: np.ndarray = None, pad_ratio: float = 0.05):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = gray.shape
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > (h_img * w_img * 0.05):
            x, y, w, h = cv2.boundingRect(c)
            pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(w_img, x + w + pad_w), min(h_img, y + h + pad_h)
            cropped_img = image[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2] if mask is not None else None
            return cropped_img, cropped_mask
    return image, mask

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

class UniversalTestDataset(Dataset):
    def __init__(self, dataset_name, base_dir, image_size=256):
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.image_size = image_size
        self.samples = []
        
        if dataset_name == 'cheng_figshare':
            csv_path = os.path.join(base_dir, 'cheng_figshare_metadata.csv')
            df = pd.read_csv(csv_path)
            test_df = df[df['split'] == 'test'].reset_index(drop=True)
            class_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
            for _, row in test_df.iterrows():
                img_rel = str(row['image_path']).replace('\\', '/')
                mask_rel = str(row['mask_path']).replace('\\', '/')
                self.samples.append({
                    'img_path': os.path.join(base_dir, img_rel),
                    'mask_path': os.path.join(base_dir, mask_rel),
                    'label': class_map[row['class_name']],
                    'class_name': row['class_name'],
                    'has_mask': True
                })
        elif dataset_name == 'brisc2025':
            csv_path = os.path.join(base_dir, 'manifest.csv')
            df = pd.read_csv(csv_path)
            seg_masks = df[(df['task'] == 'segmentation') & (df['is_mask'] == True)]
            mask_lookup = {row['filename']: str(row['relative_path']).replace('\\', '/') for _, row in seg_masks.iterrows()}
            test_df = df[(df['task'] == 'classification') & (df['split'] == 'test')].reset_index(drop=True)
            class_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'no_tumor': 3}
            for _, row in test_df.iterrows():
                fname = row['filename']
                rel_p = str(row['relative_path']).replace('\\', '/')
                m_path = os.path.join(base_dir, mask_lookup[fname]) if fname in mask_lookup else None
                self.samples.append({
                    'img_path': os.path.join(base_dir, rel_p),
                    'mask_path': m_path,
                    'label': class_map[row['tumor_label']],
                    'class_name': row['tumor_label'],
                    'has_mask': (m_path is not None) or (row['tumor_label'] == 'no_tumor')
                })
        else: # nickparvar_mri
            test_dir = os.path.join(base_dir, 'test')
            class_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}
            for c in ['glioma', 'meningioma', 'pituitary', 'notumor']:
                c_dir = os.path.join(test_dir, c)
                if os.path.exists(c_dir):
                    for fname in sorted(os.listdir(c_dir)):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            f_path = os.path.join(c_dir, fname)
                            base_name = os.path.splitext(fname)[0]
                            m_path = os.path.join(base_dir, "masks", "test", c, f"{base_name}_mask.png")
                            has_m = os.path.exists(m_path) or (c == 'notumor')
                            self.samples.append({
                                'img_path': f_path,
                                'mask_path': m_path if os.path.exists(m_path) else None,
                                'label': class_map[c],
                                'class_name': 'no_tumor' if c == 'notumor' else c,
                                'has_mask': has_m
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = cv2.imread(item['img_path'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            
        mask = None
        if item['mask_path'] and os.path.exists(item['mask_path']):
            mask = cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE)
        elif item['class_name'] in ['no_tumor', 'notumor']:
            mask = np.zeros_like(image, dtype=np.uint8)
            
        image, mask = crop_brain_contour(image, mask)
        image = apply_clahe(image)
        
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
        img_rgb = np.stack([image, image, image], axis=-1).astype(np.float32) / 255.0
        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.tensor(img_norm.transpose(2, 0, 1), dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'label': item['label'],
            'has_mask': item['has_mask']
        }

class FastCachedDataset(Dataset):
    """Multithreaded pre-cached dataset."""
    def __init__(self, raw_ds, desc="", max_workers=8):
        print(f"Pre-caching {desc} ({len(raw_ds)} slices) using {max_workers} threads...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.items = list(executor.map(lambda i: raw_ds[i], range(len(raw_ds))))
        print(f"  -> Pre-cached {len(self.items)} items in {time.time() - t0:.1f}s!", flush=True)
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        return self.items[idx]

class DualHeadDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnext101_32x8d",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            aux_params=dict(pooling='avg', dropout=0.3, classes=num_classes)
        )
    def forward(self, x):
        mask_logits, class_logits = self.model(x)
        return torch.sigmoid(mask_logits), class_logits

def compute_batch_dice(pred_probs, target_masks, threshold=0.5, smooth=1e-6):
    preds = (pred_probs > threshold).float()
    intersection = (preds * target_masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target_masks.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def compute_batch_iou(pred_probs, target_masks, threshold=0.5, smooth=1e-6):
    preds = (pred_probs > threshold).float()
    intersection = (preds * target_masks).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + target_masks.sum(dim=(1, 2, 3))
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

@torch.inference_mode()
def evaluate_pytorch_cross_domain(model, source_cohort, target_cohort, dataloader):
    model.eval()
    all_dices = []
    all_ious = []
    all_preds = []
    all_targets = []
    
    source_is_3 = (source_cohort == 'cheng_figshare')
    target_is_3 = (target_cohort == 'cheng_figshare')
    
    total_batches = len(dataloader)
    for b_idx, batch in enumerate(dataloader):
        images = batch['image']
        masks  = batch['mask']
        labels = batch['label']
        has_mask = batch['has_mask']
        
        pred_mask_probs, pred_class_logits = model(images)
        
        # Segmentation evaluation
        if has_mask.sum() > 0:
            sub_preds = pred_mask_probs[has_mask]
            sub_masks = masks[has_mask]
            dices = compute_batch_dice(sub_preds, sub_masks)
            ious  = compute_batch_iou(sub_preds, sub_masks)
            all_dices.extend(dices.cpu().numpy().tolist())
            all_ious.extend(ious.cpu().numpy().tolist())
            
        # Classification evaluation
        if source_is_3 and not target_is_3:
            valid_idx = (labels < 3)
            if valid_idx.sum() > 0:
                cls_preds = torch.argmax(pred_class_logits[valid_idx], dim=1)
                all_preds.extend(cls_preds.cpu().numpy().tolist())
                all_targets.extend(labels[valid_idx].cpu().numpy().tolist())
        elif not source_is_3 and target_is_3:
            restricted_logits = pred_class_logits[:, :3]
            cls_preds = torch.argmax(restricted_logits, dim=1)
            all_preds.extend(cls_preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
        else:
            cls_preds = torch.argmax(pred_class_logits, dim=1)
            all_preds.extend(cls_preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            
        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
            print(f"    -> Progress: batch [{b_idx+1}/{total_batches}] evaluated...", flush=True)
            
    mean_dice = float(np.mean(all_dices)) if len(all_dices) > 0 else 0.0
    mean_iou  = float(np.mean(all_ious)) if len(all_ious) > 0 else 0.0
    acc       = float(accuracy_score(all_targets, all_preds) * 100.0) if len(all_targets) > 0 else 0.0
    f1        = float(f1_score(all_targets, all_preds, average='macro') * 100.0) if len(all_targets) > 0 else 0.0
    
    return {
        'dice': round(mean_dice, 4),
        'iou':  round(mean_iou, 4),
        'acc':  round(acc, 2),
        'f1':   round(f1, 2)
    }

def main():
    print("=" * 80, flush=True)
    print("RUNNING CERTIFIED REAL-DATA EVALUATION FOR DeepLabV3Plus-ResNeXt", flush=True)
    print("=" * 80, flush=True)
    
    dataset_dirs = {
        'cheng_figshare': r"g:\My Drive\Brain Tumor Detection\datasets\cheng figshare last",
        'brisc2025':      r"g:\My Drive\Brain Tumor Detection\datasets\brisc2025",
        'nickparvar_mri': r"g:\My Drive\Brain Tumor Detection\datasets\nickparvar_mri"
    }
    
    # Pre-cache datasets
    test_loaders = {}
    for d_key, d_dir in dataset_dirs.items():
        raw_ds = UniversalTestDataset(d_key, d_dir, image_size=IMAGE_SIZE)
        cached_ds = FastCachedDataset(raw_ds, desc=d_key, max_workers=8)
        loader = DataLoader(cached_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loaders[d_key] = loader
        print(f"Ready: DataLoader for {d_key} ({len(cached_ds)} samples)\n", flush=True)

    ckpt_dir = r"g:\My Drive\Brain Tumor Detection\trained model input"
    cohort_keys = ["cheng_figshare", "brisc2025", "nickparvar_mri"]
    
    new_records = []
    
    for s_cohort in cohort_keys:
        num_classes = 3 if s_cohort == 'cheng_figshare' else 4
        ckpt_name = f"best_DeepLabV3Plus-ResNeXt_{s_cohort}.pth"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        
        print(f"\n[{s_cohort.upper()}] Loading Checkpoint: {ckpt_name}", flush=True)
        t_load = time.time()
        model = DualHeadDeepLabV3Plus(num_classes=num_classes)
        state = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded successfully in {time.time() - t_load:.1f}s", flush=True)
        
        for t_cohort in cohort_keys:
            is_in = (s_cohort == t_cohort)
            eval_type = "In-Domain" if is_in else "Cross-Domain"
            loader = test_loaders[t_cohort]
            
            print(f"  Starting evaluation: {s_cohort} -> {t_cohort} ({eval_type})...", flush=True)
            t0 = time.time()
            metrics = evaluate_pytorch_cross_domain(model, s_cohort, t_cohort, loader)
            elapsed = time.time() - t0
            
            rec = {
                'Model Name': 'DeepLabV3Plus-ResNeXt',
                'Source Domain (Train)': s_cohort,
                'Target Domain (Test)': t_cohort,
                'Evaluation Type': eval_type,
                'Dice Score': metrics['dice'],
                'Mean IoU': metrics['iou'],
                'Accuracy (%)': metrics['acc'],
                'Macro F1 (%)': metrics['f1'],
                'Eval Time (s)': round(elapsed, 2)
            }
            new_records.append(rec)
            
            tag = "[IN-DOMAIN]" if is_in else "[TRANSFER] "
            print(f"  {tag} {s_cohort:>15} -> {t_cohort:<15} | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | Acc: {metrics['acc']:5.2f}% | F1: {metrics['f1']:5.2f}% ({elapsed:.1f}s)", flush=True)

    out_csv = r"g:\My Drive\Brain Tumor Detection\cross_domain_outputs\deeplabv3plus_real_results.csv"
    pd.DataFrame(new_records).to_csv(out_csv, index=False)
    print(f"\n=======================================================", flush=True)
    print(f"ALL 9 CERTIFIED EVALUATIONS COMPLETED AND SAVED TO:", flush=True)
    print(f"{out_csv}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == '__main__':
    main()
