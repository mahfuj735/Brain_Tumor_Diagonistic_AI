import os
import sys
import time
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
DATASET_DIR = "datasets/nickparvar_mri"
REMOVE_ORIGINAL_JPG = True   # True: removes .jpg after .png is verified (avoids duplicate images in ImageFolder)
MAX_WORKERS = 16            # Multi-threading workers for fast I/O on Google Drive

def convert_single_image(jpg_path, remove_jpg=REMOVE_ORIGINAL_JPG):
    """
    Converts a single .jpg image to .png losslessly.
    Verifies the output .png before optionally removing the .jpg.
    """
    png_path = os.path.splitext(jpg_path)[0] + ".png"
    try:
        # If PNG already exists and is non-empty, skip re-saving
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

def main():
    print("=" * 60)
    print("Nickparvar MRI: Batch JPG to Lossless PNG Converter")
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

    # 3. Final verification
    remaining_jpg = sum(1 for _, _, files in os.walk(DATASET_DIR) for f in files if f.lower().endswith((".jpg", ".jpeg")))
    total_png = sum(1 for _, _, files in os.walk(DATASET_DIR) for f in files if f.lower().endswith(".png"))
    print("\nFinal Status:")
    print(f"  - Remaining JPG files: {remaining_jpg}")
    print(f"  - Total PNG files    : {total_png}")
    print("=" * 60)

if __name__ == "__main__":
    main()
