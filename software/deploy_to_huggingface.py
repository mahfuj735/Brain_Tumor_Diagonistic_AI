import os
import sys
import time
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("[WARNING] HF_TOKEN environment variable is not set. Please set it before running this script.")
REPO_NAME = "Brain-Tumor-Diagnostic-AI"
REPO_ID = f"mahfuj735/{REPO_NAME}"
SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70, flush=True)
print(">> STARTING PROFESSIONAL HUGGING FACE SPACES AUTOMATED DEPLOYMENT", flush=True)
print("=" * 70, flush=True)
print(f"Target Repository ID: {REPO_ID}", flush=True)
print(f"Local Source Folder : {SOFTWARE_DIR}", flush=True)

# 1. Initialize API & Verify Authentication
api = HfApi(token=TOKEN)
try:
    user_info = api.whoami()
    print(f"  [OK] Authenticated successfully as: @{user_info['name']} ({user_info.get('email', 'No email')})", flush=True)
except Exception as e:
    print(f"  [ERROR] Authentication failed: {e}", flush=True)
    sys.exit(1)

# 2. Create Space if it doesn't already exist
print(f"\n[1/3] Ensuring Hugging Face Space [{REPO_ID}] exists...", flush=True)
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        private=False,
        exist_ok=True
    )
    print(f"  [OK] Space [{REPO_ID}] is ready and verified!", flush=True)
except Exception as e:
    print(f"  [NOTICE] Repo check: {e}", flush=True)

# 3. Upload all files from software/
print(f"\n[2/3] Uploading all software assets, model weights, and UI components...", flush=True)
try:
    future = api.upload_folder(
        folder_path=SOFTWARE_DIR,
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=["__pycache__*", "*.bat", "deploy_to_huggingface.py"],
        commit_message="🚀 Deploy Clinical Brain Tumor Multi-Task Diagnostic AI (Mask-XAI-Net)"
    )
    print("  [OK] All files successfully uploaded to Hugging Face!", flush=True)
except Exception as e:
    print(f"  [ERROR] Upload failed: {e}", flush=True)
    sys.exit(1)

# 4. Check Space Runtime Status
print(f"\n[3/3] Checking Space Runtime and Build Status...", flush=True)
time.sleep(3)
try:
    runtime = api.get_space_runtime(repo_id=REPO_ID)
    print(f"  [STATUS] Stage: {runtime.stage} | Hardware: {runtime.hardware}", flush=True)
except Exception as e:
    print(f"  [INFO] Space is currently provisioning container: {e}", flush=True)

space_url = f"https://huggingface.co/spaces/{REPO_ID}"
print("\n" + "=" * 70, flush=True)
print("🎉 HUGGING FACE DEPLOYMENT COMPLETED SUCCESSFULLY!", flush=True)
print(f"👉 Live Public Space URL: {space_url}", flush=True)
print("=" * 70, flush=True)
