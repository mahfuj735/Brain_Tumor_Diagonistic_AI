import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

RANDOM_SEED = 42

def split_cheng_figshare():
    print("\n" + "=" * 60)
    print("1. Processing Cheng Figshare -> train, val, test")
    print("=" * 60)
    base_dir = r"datasets/cheng_figshare"
    csv_path = os.path.join(base_dir, "cheng_figshare_metadata.csv")
    backup_csv = os.path.join(base_dir, "cheng_figshare_metadata_pre_val_backup.csv")
    
    df = pd.read_csv(csv_path)
    if not os.path.exists(backup_csv):
        shutil.copy2(csv_path, backup_csv)
    
    # We use Fold 4 for validation! Fold 1,2,3 for train. Fold 5 for test.
    val_slices = df[df['fold'] == 4]
    print(f"Moving Fold 4 ({len(val_slices)} slices) to 'val' split...")
    
    classes = ['glioma', 'meningioma', 'pituitary']
    for cat in ['images', 'masks']:
        for c in classes:
            os.makedirs(os.path.join(base_dir, "val", cat, c), exist_ok=True)
            
    updated_records = []
    for _, row in val_slices.iterrows():
        slice_id = row['slice_id']
        cls = row['class_name']
        
        # Source (currently in train)
        src_img = os.path.join(base_dir, "train", "images", cls, f"{slice_id}.png")
        src_mask = os.path.join(base_dir, "train", "masks", cls, f"{slice_id}.png")
        
        # Target (val)
        dst_img = os.path.join(base_dir, "val", "images", cls, f"{slice_id}.png")
        dst_mask = os.path.join(base_dir, "val", "masks", cls, f"{slice_id}.png")
        
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_mask) and not os.path.exists(dst_mask):
            shutil.move(src_mask, dst_mask)
            
    # Update dataframe
    for _, row in df.iterrows():
        rec = row.to_dict()
        fold = rec['fold']
        if fold == 5:
            rec['split'] = 'test'
            rec['image_path'] = f"test/images/{rec['class_name']}/{rec['slice_id']}.png"
            rec['mask_path'] = f"test/masks/{rec['class_name']}/{rec['slice_id']}.png"
        elif fold == 4:
            rec['split'] = 'val'
            rec['image_path'] = f"val/images/{rec['class_name']}/{rec['slice_id']}.png"
            rec['mask_path'] = f"val/masks/{rec['class_name']}/{rec['slice_id']}.png"
        else:
            rec['split'] = 'train'
            rec['image_path'] = f"train/images/{rec['class_name']}/{rec['slice_id']}.png"
            rec['mask_path'] = f"train/masks/{rec['class_name']}/{rec['slice_id']}.png"
        updated_records.append(rec)
        
    new_df = pd.DataFrame(updated_records)
    new_df.to_csv(csv_path, index=False)
    
    # Verify counts
    for s in ['train', 'val', 'test']:
        n_img = sum(len(fs) for _, _, fs in os.walk(os.path.join(base_dir, s, "images")))
        n_msk = sum(len(fs) for _, _, fs in os.walk(os.path.join(base_dir, s, "masks")))
        print(f"  Figshare {s:5s} -> Images: {n_img:4d}, Masks: {n_msk:4d}")


def split_nickparvar_mri():
    print("\n" + "=" * 60)
    print("2. Processing Nickparvar MRI -> train, val, test")
    print("=" * 60)
    base_dir = r"datasets/nickparvar_mri"
    
    # 1. Rename Testing -> test, Training -> train if needed
    if os.path.exists(os.path.join(base_dir, "Testing")) and not os.path.exists(os.path.join(base_dir, "test")):
        os.rename(os.path.join(base_dir, "Testing"), os.path.join(base_dir, "test"))
        print("Renamed 'Testing' -> 'test'")
        
    if os.path.exists(os.path.join(base_dir, "Training")) and not os.path.exists(os.path.join(base_dir, "train")):
        os.rename(os.path.join(base_dir, "Training"), os.path.join(base_dir, "train"))
        print("Renamed 'Training' -> 'train'")
        
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    for c in classes:
        os.makedirs(os.path.join(base_dir, "val", c), exist_ok=True)
        
    # Split 15% from train into val
    np.random.seed(RANDOM_SEED)
    total_val_moved = 0
    for c in classes:
        train_class_dir = os.path.join(base_dir, "train", c)
        val_class_dir = os.path.join(base_dir, "val", c)
        
        # Only move if val is currently empty
        if len(os.listdir(val_class_dir)) == 0:
            files = sorted(os.listdir(train_class_dir))
            n_val = int(len(files) * 0.15)
            val_files = np.random.choice(files, n_val, replace=False)
            
            for f in val_files:
                shutil.move(os.path.join(train_class_dir, f), os.path.join(val_class_dir, f))
                total_val_moved += 1
            print(f"  Moved {n_val:3d} files from train/{c} to val/{c}")
        else:
            print(f"  val/{c} already populated ({len(os.listdir(val_class_dir))} files)")
            
    # Verify counts
    for s in ['train', 'val', 'test']:
        n = sum(len(fs) for _, _, fs in os.walk(os.path.join(base_dir, s)))
        print(f"  Nickparvar {s:5s} -> Total files: {n:4d}")


def split_brisc2025():
    print("\n" + "=" * 60)
    print("3. Processing BRISC 2025 -> train, val, test")
    print("=" * 60)
    base_dir = r"datasets/brisc2025"
    manifest_csv = os.path.join(base_dir, "manifest.csv")
    backup_manifest = os.path.join(base_dir, "manifest_pre_val_backup.csv")
    
    df = pd.read_csv(manifest_csv)
    if not os.path.exists(backup_manifest):
        shutil.copy2(manifest_csv, backup_manifest)
        
    classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    for c in classes:
        os.makedirs(os.path.join(base_dir, "classification_task", "val", c), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "segmentation_task", "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "segmentation_task", "val", "masks"), exist_ok=True)
    
    # Check if val is already created
    train_class_rows = df[(df['task'] == 'classification') & (df['split'] == 'train')]
    
    np.random.seed(RANDOM_SEED)
    val_filenames = set()
    
    # 15% stratified from each class in classification train
    for c in classes:
        cls_rows = train_class_rows[train_class_rows['tumor_label'] == c]
        n_val = int(len(cls_rows) * 0.15)
        selected = np.random.choice(cls_rows['filename'].values, n_val, replace=False)
        val_filenames.update(selected)
        print(f"  Selected {n_val:3d} images from BRISC train/{c} for val")
        
    print(f"Total BRISC validation filenames selected: {len(val_filenames)}")
    
    # Move files in classification_task
    moved_c = 0
    for f in val_filenames:
        # Find which class it belongs to
        row = df[(df['task'] == 'classification') & (df['filename'] == f)].iloc[0]
        c = row['tumor_label']
        src = os.path.join(base_dir, "classification_task", "train", c, f)
        dst = os.path.join(base_dir, "classification_task", "val", c, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved_c += 1
            
    # Move files in segmentation_task (if it's a tumor image)
    moved_s_img = 0
    moved_s_msk = 0
    for f in val_filenames:
        src_img = os.path.join(base_dir, "segmentation_task", "train", "images", f)
        dst_img = os.path.join(base_dir, "segmentation_task", "val", "images", f)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.move(src_img, dst_img)
            moved_s_img += 1
            
        src_msk = os.path.join(base_dir, "segmentation_task", "train", "masks", f)
        dst_msk = os.path.join(base_dir, "segmentation_task", "val", "masks", f)
        if os.path.exists(src_msk) and not os.path.exists(dst_msk):
            shutil.move(src_msk, dst_msk)
            moved_s_msk += 1
            
    print(f"Moved {moved_c} classification images, {moved_s_img} seg images, {moved_s_msk} seg masks to 'val'")
    
    # Update manifest.csv
    updated_rows = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        fn = rec['filename']
        if fn in val_filenames and rec['split'] == 'train':
            rec['split'] = 'val'
            # Update relative_path
            rel = rec['relative_path']
            rec['relative_path'] = rel.replace("\\train\\", "\\val\\").replace("/train/", "/val/")
            if pd.notna(rec.get('linked_image')):
                rec['linked_image'] = rec['linked_image'].replace("\\train\\", "\\val\\").replace("/train/", "/val/")
        updated_rows.append(rec)
        
    new_df = pd.DataFrame(updated_rows)
    new_df.to_csv(manifest_csv, index=False)
    print("Updated BRISC manifest.csv saved.")
    
    # Verify counts
    for task in ['classification_task', 'segmentation_task']:
        print(f"\n[{task}]")
        for s in ['train', 'val', 'test']:
            tp = os.path.join(base_dir, task, s)
            n = sum(len(fs) for _, _, fs in os.walk(tp))
            print(f"  {task}/{s:5s} -> Total files: {n:4d}")

def main():
    print("=" * 60)
    print("CREATING UNIFIED TRAIN / VAL / TEST SPLITS ACROSS ALL 3 BENCHMARKS")
    print("=" * 60)
    split_cheng_figshare()
    split_nickparvar_mri()
    split_brisc2025()
    print("\n" + "=" * 60)
    print("ALL 3 BENCHMARKS SUCCESSFULLY REORGANIZED INTO TRAIN, VAL, TEST!")
    print("=" * 60)

if __name__ == "__main__":
    main()
