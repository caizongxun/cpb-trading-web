#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Upload V6 Models to HuggingFace (FOLDER METHOD)

Uploads entire models_v6 folder in ONE operation
Simple and efficient - no file-by-file uploads

Usage:
  Method 1: Environment Variable
    export HF_TOKEN='your_hf_write_token'
    python upload_models_v6_folder.py
  
  Method 2: Command Line Argument
    python upload_models_v6_folder.py your_hf_write_token
  
  Method 3: Interactive Input
    python upload_models_v6_folder.py
    (will prompt for token)
"""

import os
import sys
import getpass
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_REPO_ID = "zongowo111/cpb-models"
LOCAL_MODELS_DIR = "models_v6"
REPO_TYPE = "dataset"
REMOTE_DIR = "models_v6"

# ============================================================================
# UTILITIES
# ============================================================================

def format_size(bytes_size):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"

def print_header():
    print("\n" + "="*80)
    print("HuggingFace Folder Uploader - models_v6")
    print("="*80 + "\n")

def print_section(title):
    print(f"\n[{title}]")
    print("-" * 80)

def get_hf_token():
    """
    Try to get HF token from multiple sources:
    1. Command line argument
    2. Environment variable HF_TOKEN
    3. Environment variable HUGGINGFACE_TOKEN
    4. Local file ~/.huggingface/token
    5. Interactive input
    """
    
    # Source 1: Command line argument
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
        if token and not token.startswith('-'):
            return token, "command line argument"
    
    # Source 2: Environment variable HF_TOKEN
    token = os.getenv('HF_TOKEN')
    if token:
        return token, "environment variable (HF_TOKEN)"
    
    # Source 3: Environment variable HUGGINGFACE_TOKEN
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        return token, "environment variable (HUGGINGFACE_TOKEN)"
    
    # Source 4: Local file
    hf_token_file = Path.home() / '.huggingface' / 'token'
    if hf_token_file.exists():
        try:
            with open(hf_token_file, 'r') as f:
                token = f.read().strip()
            if token:
                return token, "local file (~/.huggingface/token)"
        except:
            pass
    
    # Source 5: Interactive input
    print("\nNo HF_TOKEN found. Please enter your HuggingFace Write Token:")
    print("(You can get it from https://huggingface.co/settings/tokens)")
    print()
    try:
        token = getpass.getpass("HF Token > ").strip()
        if token:
            return token, "interactive input"
    except:
        pass
    
    return None, None

def main():
    print_header()
    
    # ========================================================================
    # STEP 1: Validate Environment
    # ========================================================================
    print_section("Step 1: Validate Environment")
    
    # Check directory
    print(f"Local Directory: {LOCAL_MODELS_DIR}")
    models_dir = Path(LOCAL_MODELS_DIR)
    
    if not models_dir.exists():
        print(f"  ✗ Directory does not exist!")
        sys.exit(1)
    
    files = list(models_dir.glob('*'))
    if not files:
        print(f"  ✗ Directory is empty!")
        sys.exit(1)
    
    keras_files = [f for f in files if f.suffix == '.keras']
    json_files = [f for f in files if f.suffix == '.json']
    h5_files = [f for f in files if f.suffix == '.h5']
    total_size = sum(f.stat().st_size for f in files)
    
    print(f"  ✓ Directory exists")
    print(f"    Files: {len(files)} total")
    print(f"    - .keras: {len(keras_files)}")
    print(f"    - .json: {len(json_files)}")
    print(f"    - .h5: {len(h5_files)}")
    print(f"    Size: {format_size(total_size)}")
    
    # Check HF Token
    print(f"\nHuggingFace Token")
    token, source = get_hf_token()
    
    if not token:
        print(f"  ✗ HF_TOKEN not found!")
        print(f"\n  Set token with one of:")
        print(f"    export HF_TOKEN='your_token'")
        print(f"    python upload_models_v6_folder.py 'your_token'")
        print(f"    Or save to ~/.huggingface/token")
        sys.exit(1)
    
    print(f"  ✓ Token found (source: {source})")
    
    # Check repository access
    print(f"\nRepository Access: {HF_REPO_ID}")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        repo_info = api.repo_info(
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
            token=token
        )
        print(f"  ✓ Repository accessible")
    except Exception as e:
        print(f"  ✗ Cannot access repository: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Verify token is correct")
        print(f"    2. Check token has write permissions")
        print(f"    3. Verify repository exists and is accessible")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Upload Folder
    # ========================================================================
    print_section("Step 2: Upload Folder")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        
        print(f"Source: {LOCAL_MODELS_DIR}")
        print(f"Target: {HF_REPO_ID}/{REMOTE_DIR}")
        print(f"Method: upload_folder (entire directory)")
        print()
        print(f"[上傳中] Uploading {len(files)} files ({format_size(total_size)})...")
        print(f"Please wait...\n")
        
        # Upload entire folder
        result = api.upload_folder(
            folder_path=LOCAL_MODELS_DIR,
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=REMOTE_DIR,
            token=token,
            commit_message="Batch upload V6 models (.keras format)"
        )
        
        print(f"\n✓ Upload completed successfully!")
        print(f"\nCommit URL: {result}")
        
    except Exception as e:
        print(f"\n✗ Upload failed!")
        print(f"Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Verify HF_TOKEN is correct and has write permissions")
        print(f"  2. Check network connection")
        print(f"  3. Try again: python upload_models_v6_folder.py")
        sys.exit(1)
    
    # ========================================================================
    # STEP 3: Verify Upload
    # ========================================================================
    print_section("Step 3: Verify Upload")
    
    try:
        api = HfApi(token=token)
        
        # List files in uploaded directory
        files_in_repo = api.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
            recursive=True,
            token=token
        )
        
        # Filter to our models_v6 directory
        uploaded_files = [f for f in files_in_repo if f.path.startswith(f"{REMOTE_DIR}/")]
        uploaded_keras = [f for f in uploaded_files if f.path.endswith('.keras')]
        uploaded_json = [f for f in uploaded_files if f.path.endswith('.json')]
        uploaded_h5 = [f for f in uploaded_files if f.path.endswith('.h5')]
        
        print(f"Uploaded files in {REMOTE_DIR}/:")
        print(f"  .keras files: {len(uploaded_keras)}")
        print(f"  .json files: {len(uploaded_json)}")
        print(f"  .h5 files: {len(uploaded_h5)}")
        print(f"  Total: {len(uploaded_files)} files")
        
        # Show sample files
        print(f"\nSample uploaded files:")
        for f in uploaded_files[:5]:
            print(f"  ✓ {f.path}")
        if len(uploaded_files) > 5:
            print(f"  ... and {len(uploaded_files) - 5} more")
        
    except Exception as e:
        print(f"⚠ Could not verify: {e}")
    
    # ========================================================================
    # STEP 4: Summary
    # ========================================================================
    print_section("Step 4: Summary")
    
    print(f"✓ Upload Completed Successfully!\n")
    print(f"Repository: https://huggingface.co/datasets/{HF_REPO_ID}")
    print(f"Models dir: https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{REMOTE_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nYou can now use these models directly from HuggingFace!")
    print()
    
    # ========================================================================
    # DONE
    # ========================================================================
    print("="*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
