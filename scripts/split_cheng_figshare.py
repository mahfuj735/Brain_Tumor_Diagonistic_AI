import os
import shutil
import pandas as pd
from tqdm import tqdm

BASE_DIR = r"datasets/cheng_figshare"
CSV_PATH = os.path.join(BASE_DIR, "cheng_figshare_metadata.csv")
BACKUP_CSV_PATH = os.path.join(BASE_DIR, "cheng_figshare_metadata_backup.csv")

def main():
    print("=" * 60)
    print("Reorganizing Cheng Figshare into Train (Folds 1-4) and Test (Fold 5)")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Metadata file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded metadata with {len(df)} records.")

    # Backup original CSV
    if not os.path.exists(BACKUP_CSV_PATH):
        shutil.copy2(CSV_PATH, BACKUP_CSV_PATH)
        print(f"Backed up original metadata to: {BACKUP_CSV_PATH}")

    classes = ['glioma', 'meningioma', 'pituitary']
    splits = ['train', 'test']

    # Create target directories
    for s in splits:
        for cat in ['images', 'masks']:
            for c in classes:
                os.makedirs(os.path.join(BASE_DIR, s, cat, c), exist_ok=True)
    print("Target directory structure created successfully.")

    updated_records = []
    moved_images = 0
    moved_masks = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Moving files"):
        fold = row['fold']
        split = 'test' if fold == 5 else 'train'
        cls = row['class_name']
        slice_id = row['slice_id']

        # Source paths
        old_rel_img = row['image_path']
        old_rel_mask = row['mask_path']
        
        old_full_img = os.path.join(BASE_DIR, old_rel_img)
        old_full_mask = os.path.join(BASE_DIR, old_rel_mask)

        # Target paths
        new_rel_img = os.path.join(split, "images", cls, f"{slice_id}.png").replace("\\", "/")
        new_rel_mask = os.path.join(split, "masks", cls, f"{slice_id}.png").replace("\\", "/")

        new_full_img = os.path.join(BASE_DIR, new_rel_img)
        new_full_mask = os.path.join(BASE_DIR, new_rel_mask)

        # Move Image if not already in target
        if os.path.exists(old_full_img) and not os.path.exists(new_full_img):
            shutil.move(old_full_img, new_full_img)
            moved_images += 1
        elif os.path.exists(new_full_img):
            pass # Already moved
        else:
            print(f"Warning: Missing image {old_full_img}")

        # Move Mask if not already in target
        if os.path.exists(old_full_mask) and not os.path.exists(new_full_mask):
            shutil.move(old_full_mask, new_full_mask)
            moved_masks += 1
        elif os.path.exists(new_full_mask):
            pass # Already moved
        else:
            print(f"Warning: Missing mask {old_full_mask}")

        # Record update
        rec = row.to_dict()
        rec['split'] = split
        rec['image_path'] = new_rel_img
        rec['mask_path'] = new_rel_mask
        updated_records.append(rec)

    # Save updated CSV
    new_df = pd.DataFrame(updated_records)
    new_df.to_csv(CSV_PATH, index=False)
    print(f"\nUpdated metadata saved to: {CSV_PATH}")

    # Remove old empty directories if they are empty
    for cat in ['images', 'masks']:
        cat_dir = os.path.join(BASE_DIR, cat)
        if os.path.exists(cat_dir):
            try:
                shutil.rmtree(cat_dir)
                print(f"Cleaned up legacy empty directory: {cat_dir}")
            except Exception as e:
                print(f"Could not remove {cat_dir}: {e}")

    # Verification
    print("\n--- Final Verification ---")
    train_imgs = sum(len(files) for _, _, files in os.walk(os.path.join(BASE_DIR, "train", "images")))
    train_msks = sum(len(files) for _, _, files in os.walk(os.path.join(BASE_DIR, "train", "masks")))
    test_imgs = sum(len(files) for _, _, files in os.walk(os.path.join(BASE_DIR, "test", "images")))
    test_msks = sum(len(files) for _, _, files in os.walk(os.path.join(BASE_DIR, "test", "masks")))

    print(f"Train Images: {train_imgs}, Train Masks: {train_msks} (Expected: 2421)")
    print(f"Test Images:  {test_imgs}, Test Masks:  {test_msks} (Expected: 643)")
    print(f"Total Images: {train_imgs + test_imgs}, Total Masks: {train_msks + test_msks} (Expected: 3064)")
    print("Done!")

if __name__ == "__main__":
    main()
