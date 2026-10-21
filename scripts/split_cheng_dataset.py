import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = "datasets/cheng_figshare"
METADATA_CSV = os.path.join(DATASET_DIR, "cheng_figshare_metadata.csv")
BACKUP_CSV = os.path.join(DATASET_DIR, "cheng_figshare_metadata_backup.csv")

def main():
    print("=" * 60)
    print("Cheng Figshare: Patient-Stratified Train / Test Splitter")
    print(f"Target Directory: {DATASET_DIR}")
    print("=" * 60)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] Metadata file not found: {METADATA_CSV}")
        return

    # 1. Load metadata
    df = pd.read_csv(METADATA_CSV)
    print(f"Total slices in metadata: {len(df)}")
    print(f"Total unique patients: {df['patient_id'].nunique()}")

    # Backup metadata
    if not os.path.exists(BACKUP_CSV):
        shutil.copy2(METADATA_CSV, BACKUP_CSV)
        print(f"[INFO] Backed up original metadata to: {BACKUP_CSV}")

    # 2. Define Split mapping based on official 5-Fold CV:
    # Folds 1, 2, 3, 4 -> Training (~79%)
    # Fold 5          -> Testing (~21%)
    # This guarantees 0% patient leakage between Train and Test!
    def get_split(fold):
        return "Testing" if fold == 5 else "Training"

    df["split"] = df["fold"].apply(get_split)
    print("\nSplit Distribution:")
    print(df["split"].value_counts())

    train_pids = set(df[df["split"] == "Training"]["patient_id"])
    test_pids = set(df[df["split"] == "Testing"]["patient_id"])
    overlap = train_pids & test_pids
    print(f"Train patients: {len(train_pids)}, Test patients: {len(test_pids)}")
    print(f"Patient overlap between Train and Test: {len(overlap)} (Zero Leakage: {len(overlap) == 0})")
    assert len(overlap) == 0, "Patient overlap detected!"

    # 3. Create destination folder structures
    classes = df["class_name"].unique()
    for split in ["Training", "Testing"]:
        for category in ["images", "masks"]:
            for c in classes:
                os.makedirs(os.path.join(DATASET_DIR, split, category, c), exist_ok=True)

    # 4. Move files carefully
    print("\nMoving files into Training and Testing structures...")
    updated_image_paths = []
    updated_mask_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting Dataset"):
        split = row["split"]
        class_name = row["class_name"]
        filename = f"{row['slice_id']}.png"

        # Source paths
        old_img_rel = row["image_path"]
        old_mask_rel = row["mask_path"]
        
        old_img_full = os.path.join(DATASET_DIR, old_img_rel)
        old_mask_full = os.path.join(DATASET_DIR, old_mask_rel)

        # Target relative paths
        new_img_rel = os.path.join(split, "images", class_name, filename).replace("\\", "/")
        new_mask_rel = os.path.join(split, "masks", class_name, filename).replace("\\", "/")

        new_img_full = os.path.join(DATASET_DIR, new_img_rel)
        new_mask_full = os.path.join(DATASET_DIR, new_mask_rel)

        # Move image
        if os.path.exists(old_img_full) and old_img_full != new_img_full:
            shutil.move(old_img_full, new_img_full)

        # Move mask
        if os.path.exists(old_mask_full) and old_mask_full != new_mask_full:
            shutil.move(old_mask_full, new_mask_full)

        updated_image_paths.append(new_img_rel)
        updated_mask_paths.append(new_mask_rel)

    # 5. Clean up old empty source directories if any
    for category in ["images", "masks"]:
        cat_dir = os.path.join(DATASET_DIR, category)
        if os.path.exists(cat_dir):
            try:
                # Only remove if empty or contains only empty subfolders
                for c in classes:
                    c_dir = os.path.join(cat_dir, c)
                    if os.path.exists(c_dir) and not os.listdir(c_dir):
                        os.rmdir(c_dir)
                if not os.listdir(cat_dir):
                    os.rmdir(cat_dir)
            except Exception:
                pass

    # 6. Update metadata CSV
    df["image_path"] = updated_image_paths
    df["mask_path"] = updated_mask_paths

    # Reorder columns nicely
    cols = ["slice_id", "patient_id", "class_id", "class_name", "split", "fold", "image_path", "mask_path", "tumor_area_pixels", "width", "height"]
    df = df[cols]
    df.to_csv(METADATA_CSV, index=False)
    print(f"\n[OK] Updated {METADATA_CSV} with new split paths!")

    # 7. Audit & Verification
    print("\n" + "=" * 60)
    print("FINAL AUDIT VERIFICATION")
    print("=" * 60)

    for split in ["Training", "Testing"]:
        img_count = sum(len(files) for _, _, files in os.walk(os.path.join(DATASET_DIR, split, "images")))
        mask_count = sum(len(files) for _, _, files in os.walk(os.path.join(DATASET_DIR, split, "masks")))
        print(f"  {split}:")
        print(f"    - Images: {img_count}")
        print(f"    - Masks : {mask_count}")
        for c in classes:
            c_img = len(os.listdir(os.path.join(DATASET_DIR, split, "images", c)))
            c_mask = len(os.listdir(os.path.join(DATASET_DIR, split, "masks", c)))
            print(f"      * {c}: {c_img} images, {c_mask} masks")

    print("=" * 60)
    print("SUCCESS: cheng_figshare successfully split into Training and Testing!")

if __name__ == "__main__":
    main()
