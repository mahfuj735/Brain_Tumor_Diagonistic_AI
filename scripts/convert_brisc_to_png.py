import os
import sys
import glob
import time
import shutil
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DATASET_DIR = "datasets/brisc2025"
REMOVE_ORIGINAL_JPG = True   # True: removes .jpg after .png is verified (avoids duplicate images in ImageFolder)
MAX_WORKERS = 16            # Multi-threading workers for fast I/O on Google Drive

def convert_single_image(jpg_path, remove_jpg=REMOVE_ORIGINAL_JPG):
    """
    Converts a single .jpg image to .png losslessly.
    Verifies the output .png before optionally removing the .jpg.
    """
    png_path = os.path.splitext(jpg_path)[0] + ".png"
    try:
        # If PNG already exists and is non-empty, skip conversion
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            with Image.open(jpg_path) as img:
                img.save(png_path, format="PNG", optimize=False)
        
        # Verify PNG exists and has non-zero size
        if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            if remove_jpg and os.path.exists(jpg_path):
                os.remove(jpg_path)
            return True, jpg_path, None
        else:
            return False, jpg_path, "PNG file was not created properly or is 0 bytes."
    except Exception as e:
        return False, jpg_path, str(e)

def update_manifest():
    """
    Updates manifest.csv and manifest.json to reflect .png extensions.
    Creates a backup of the original manifest before modifying.
    """
    manifest_csv = os.path.join(DATASET_DIR, "manifest.csv")
    if os.path.exists(manifest_csv):
        backup_csv = os.path.join(DATASET_DIR, "manifest_backup_jpg.csv")
        if not os.path.exists(backup_csv):
            shutil.copy2(manifest_csv, backup_csv)
            print(f"[INFO] Backed up original manifest to: {backup_csv}")
        
        df = pd.read_csv(manifest_csv)
        if "relative_path" in df.columns:
            df["relative_path"] = df["relative_path"].str.replace(".jpg", ".png", case=False)
        if "filename" in df.columns:
            df["filename"] = df["filename"].str.replace(".jpg", ".png", case=False)
        if "linked_image" in df.columns:
            df["linked_image"] = df["linked_image"].str.replace(".jpg", ".png", case=False)
        
        df.to_csv(manifest_csv, index=False)
        print(f"[OK] Updated {manifest_csv} with .png file extensions!")

def main():
    print("=" * 60)
    print("BRISC 2025: Batch JPG to Lossless PNG Converter")
    print(f"Target Directory: {DATASET_DIR}")
    print(f"Remove JPG after conversion: {REMOVE_ORIGINAL_JPG}")
    print(f"Worker Threads: {MAX_WORKERS}")
    print("=" * 60)

    # 1. Search for all .jpg / .jpeg files
    print("\nScanning for JPG files...")
    jpg_files = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                jpg_files.append(os.path.join(root, f))
    
    total_files = len(jpg_files)
    print(f"Found {total_files} JPG files to convert.")
    
    if total_files == 0:
        print("No JPG files found. Checking if files are already in PNG format...")
        png_count = sum(len(files) for _, _, files in os.walk(DATASET_DIR) if any(f.lower().endswith('.png') for f in files))
        print(f"Total PNG files in dataset: {png_count}")
        return

    # 2. Run multi-threaded conversion
    t0 = time.time()
    success_count = 0
    fail_count = 0
    errors = []

    print("\nConverting images in parallel...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(convert_single_image, f, REMOVE_ORIGINAL_JPG): f for f in jpg_files}
        
        with tqdm(total=total_files, desc="Converting to PNG", unit="img") as pbar:
            for future in as_completed(futures):
                success, path, err = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    errors.append((path, err))
                pbar.update(1)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Conversion Completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} mins)!")
    print(f"[OK] Successfully converted : {success_count} files")
    print(f"[FAIL] Failed               : {fail_count} files")
    
    if errors:
        print("\nErrors encountered:")
        for path, err in errors[:10]:
            print(f"  - {path}: {err}")
    
    # 3. Update manifest
    print("\nUpdating dataset manifest...")
    update_manifest()

    # 4. Final verification
    remaining_jpg = sum(1 for _, _, files in os.walk(DATASET_DIR) for f in files if f.lower().endswith((".jpg", ".jpeg")))
    total_png = sum(1 for _, _, files in os.walk(DATASET_DIR) for f in files if f.lower().endswith(".png"))
    print("\nFinal Status:")
    print(f"  - Remaining JPG files: {remaining_jpg}")
    print(f"  - Total PNG files    : {total_png}")
    print("=" * 60)

if __name__ == "__main__":
    main()
