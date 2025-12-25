#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Upload for Google Colab

This script is designed to work directly in Google Colab
No need to clone repository or set up environment

Usage in Colab:
  !pip install --upgrade huggingface_hub
  !git clone https://github.com/caizongxun/cpb-trading-web.git
  %cd cpb-trading-web
  !python colab_upload.py hf_your_token_here

Or using secrets:
  from google.colab import userdata
  token = userdata.get('HF_TOKEN')
  # Then run this script with token
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def format_size(bytes_size):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"

def print_header():
    print("\n" + "="*80)
    print("HuggingFace Upload - Google Colab Version")
    print("="*80 + "\n")

def print_section(title):
    print(f"\n[{title}]")
    print("-" * 80)

def get_colab_token():
    """
    Try to get token from Colab Secrets first, then command line
    """
    # Method 1: Try Colab Secrets
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token, "Colab Secrets (HF_TOKEN)"
    except:
        pass
    
    # Method 2: Command line argument
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
        if token and not token.startswith('-'):
            return token, "command line argument"
    
    # Method 3: Environment variable
    token = os.getenv('HF_TOKEN')
    if token:
        return token, "environment variable (HF_TOKEN)"
    
    return None, None

def main():
    print_header()
    
    # Check if we're in Colab
    in_colab = False
    try:
        from google.colab import drive
        in_colab = True
        print("✓ Running in Google Colab")
    except:
        print("Note: Not running in Google Colab (running locally instead)")
    
    # ========================================================================
    # STEP 1: Locate models_v6 directory
    # ========================================================================
    print_section("Step 1: Locate Models Directory")
    
    # Try multiple possible locations
    possible_locations = [
        Path("models_v6"),
        Path("cpb-trading-web/models_v6"),
        Path("/content/cpb-trading-web/models_v6"),
        Path("/root/cpb-trading-web/models_v6"),
    ]
    
    models_dir = None
    for loc in possible_locations:
        if loc.exists():
            models_dir = loc
            print(f"  ✓ Found models_v6 at: {loc.absolute()}")
            break
    
    if not models_dir:
        print(f"  ✗ models_v6 directory not found!")
        print(f"\n  Make sure you:")
        print(f"    1. Cloned the repository: !git clone https://github.com/caizongxun/cpb-trading-web.git")
        print(f"    2. Changed directory: %cd cpb-trading-web")
        print(f"    3. Have converted models: !python convert_h5_to_keras_fixed.py")
        sys.exit(1)
    
    # Count files
    files = list(models_dir.glob('*'))
    if not files:
        print(f"  ✗ models_v6 directory is empty!")
        sys.exit(1)
    
    keras_files = [f for f in files if f.suffix == '.keras']
    json_files = [f for f in files if f.suffix == '.json']
    h5_files = [f for f in files if f.suffix == '.h5']
    total_size = sum(f.stat().st_size for f in files)
    
    print(f"  Files in models_v6:")
    print(f"    - .keras: {len(keras_files)}")
    print(f"    - .json: {len(json_files)}")
    print(f"    - .h5: {len(h5_files)}")
    print(f"    Total: {len(files)} files ({format_size(total_size)})")
    
    if len(keras_files) == 0:
        print(f"\n  ⚠ No .keras files found!")
        print(f"  Run conversion first: !python convert_h5_to_keras_fixed.py")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Get HF Token
    # ========================================================================
    print_section("Step 2: HuggingFace Token")
    
    token, source = get_colab_token()
    
    if not token:
        print(f"  ✗ HF_TOKEN not found!")
        print(f"\n  Set token with one of:")
        print(f"    Option 1 (Recommended): Use Colab Secrets")
        print(f"      - Click on Secrets icon (left sidebar)")
        print(f"      - Add new secret: HF_TOKEN")
        print(f"      - Paste your HuggingFace token")
        print(f"\n    Option 2: Pass as command line argument")
        print(f"      !python colab_upload.py hf_your_token_here")
        print(f"\n    Option 3: Set environment variable")
        print(f"      !export HF_TOKEN='hf_your_token_here'")
        sys.exit(1)
    
    print(f"  ✓ Token found (source: {source})")
    print(f"    Length: {len(token)} characters")
    print(f"    Preview: {token[:20]}...")
    
    # ========================================================================
    # STEP 3: Test Repository Access
    # ========================================================================
    print_section("Step 3: Test Repository Access")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        print(f"  Checking repository: zongowo111/cpb-models")
        repo_info = api.repo_info(
            repo_id="zongowo111/cpb-models",
            repo_type="dataset",
            token=token
        )
        print(f"  ✓ Repository accessible")
    except Exception as e:
        print(f"  ✗ Cannot access repository: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Verify token is correct")
        print(f"    2. Make sure token has WRITE permission")
        print(f"    3. Get token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: Upload Folder
    # ========================================================================
    print_section("Step 4: Upload Folder")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        
        print(f"  Source: {models_dir.absolute()}")
        print(f"  Target: zongowo111/cpb-models/models_v6")
        print(f"  Method: upload_folder (entire directory)")
        print(f"\n  [上傳中] Uploading {len(files)} files ({format_size(total_size)})")
        print(f"  Please wait...\n")
        
        # Upload entire folder
        result = api.upload_folder(
            folder_path=str(models_dir),
            repo_id="zongowo111/cpb-models",
            repo_type="dataset",
            path_in_repo="models_v6",
            token=token,
            commit_message="Batch upload V6 models (.keras format) from Colab"
        )
        
        print(f"\n  ✓ Upload completed successfully!")
        print(f"\n  Commit URL: {result}")
        
    except Exception as e:
        print(f"\n  ✗ Upload failed!")
        print(f"  Error: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Check network connection")
        print(f"    2. Verify token has write permissions")
        print(f"    3. Try again: !python colab_upload.py hf_your_token")
        sys.exit(1)
    
    # ========================================================================
    # STEP 5: Verify Upload
    # ========================================================================
    print_section("Step 5: Verify Upload")
    
    try:
        api = HfApi(token=token)
        
        # List uploaded files
        files_in_repo = api.list_repo_tree(
            repo_id="zongowo111/cpb-models",
            repo_type="dataset",
            recursive=True,
            token=token
        )
        
        # Filter to models_v6 directory
        uploaded = [f for f in files_in_repo if f.path.startswith("models_v6/")]
        keras_uploaded = [f for f in uploaded if f.path.endswith('.keras')]
        json_uploaded = [f for f in uploaded if f.path.endswith('.json')]
        
        print(f"  Uploaded files in models_v6/:")
        print(f"    .keras files: {len(keras_uploaded)}")
        print(f"    .json files: {len(json_uploaded)}")
        print(f"    Total: {len(uploaded)} files")
        
        if len(keras_uploaded) == 54 and len(json_uploaded) == 54:
            print(f"\n  ✓ All files uploaded successfully!")
        else:
            print(f"\n  ⚠ Upload completed but file count mismatch")
            print(f"    Expected: 54 .keras + 54 .json")
            print(f"    Actual: {len(keras_uploaded)} .keras + {len(json_uploaded)} .json")
        
    except Exception as e:
        print(f"  ⚠ Could not verify: {e}")
    
    # ========================================================================
    # STEP 6: Summary
    # ========================================================================
    print_section("Step 6: Summary")
    
    print(f"  ✓ Upload Completed Successfully!\n")
    print(f"  Repository: https://huggingface.co/datasets/zongowo111/cpb-models")
    print(f"  Models dir: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v6")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"  You can now use these models directly from HuggingFace!")
    print(f"  Example usage:")
    print(f"    from huggingface_hub import hf_hub_download")
    print(f"    model_path = hf_hub_download(")
    print(f"        repo_id='zongowo111/cpb-models',")
    print(f"        filename='models_v6/BTC_1h.keras',")
    print(f"        repo_type='dataset'")
    print(f"    )")
    print()
    print("="*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
