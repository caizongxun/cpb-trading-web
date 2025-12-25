#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Upload V6 Models to HuggingFace

Uploads entire models_v6 directory to HF in ONE batch operation
Avoiding API rate limits and ensuring complete data consistency

Usage:
  In Colab:
    !python upload_models_v6_batch.py
  
  Locally:
    export HF_TOKEN='your_token_here'
    python upload_models_v6_batch.py

Model Format: .keras (modern TensorFlow format)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_REPO_ID = "zongowo111/cpb-models"
LOCAL_MODELS_DIR = "models_v6"
REPO_TYPE = "dataset"
REMOTE_DIR = "models_v6"
COMMIT_MESSAGE = "Batch upload V6 models (.keras format)"
MAX_RETRIES = 3
RETRY_DELAY = 5  # Initial retry delay in seconds

# ============================================================================
# UTILITIES
# ============================================================================

def print_header():
    print("\n" + "="*80)
    print("HuggingFace Batch Uploader - models_v6 Directory")
    print("="*80 + "\n")

def print_section(title):
    print(f"\n[{title}]")
    print("-" * 80)

def format_size(bytes_size):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"

def get_token_from_sources():
    """
    Try to get HF token from multiple sources:
    1. Environment variable: HF_TOKEN
    2. Colab Secrets (automatically loaded)
    3. Local file: ~/.huggingface/token
    """
    # Try environment variable
    token = os.getenv('HF_TOKEN')
    if token:
        return token, "environment variable (HF_TOKEN)"
    
    # Try Colab Secrets
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token, "Colab Secrets"
    except:
        pass
    
    # Try local file
    hf_token_file = Path.home() / '.huggingface' / 'token'
    if hf_token_file.exists():
        with open(hf_token_file, 'r') as f:
            token = f.read().strip()
        if token:
            return token, "local file (~/.huggingface/token)"
    
    return None, None

def validate_environment():
    """
    Validate that we have everything needed for upload
    """
    print_section("Environment Validation")
    
    # Check local directory
    print(f"Local Directory: {LOCAL_MODELS_DIR}")
    if not Path(LOCAL_MODELS_DIR).exists():
        print(f"  ✗ Directory does not exist!")
        return False
    
    files = list(Path(LOCAL_MODELS_DIR).glob('*'))
    if not files:
        print(f"  ✗ Directory is empty!")
        return False
    
    keras_files = [f for f in files if f.suffix == '.keras']
    json_files = [f for f in files if f.suffix == '.json']
    total_size = sum(f.stat().st_size for f in files)
    
    print(f"  ✓ Directory exists, contains {len(files)} files")
    print(f"    - .keras files: {len(keras_files)}")
    print(f"    - .json files: {len(json_files)}")
    print(f"    - Total size: {format_size(total_size)}")
    
    if len(keras_files) == 0:
        print(f"  ⚠ Warning: No .keras files found. Make sure training completed.")
        return False
    
    # Check HF Token
    print(f"\nHuggingFace Token")
    token, source = get_token_from_sources()
    
    if not token:
        print(f"  ✗ No HF token found!")
        print(f"    Set HF_TOKEN environment variable or configure Colab Secrets")
        return False
    
    print(f"  ✓ Token found (source: {source})")
    
    # Check repository access
    print(f"\nHuggingFace Repository: {HF_REPO_ID}")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        # Try to get repo info
        repo_info = api.repo_info(
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
            token=token
        )
        print(f"  ✓ Repository accessible")
        print(f"    - Type: {REPO_TYPE}")
        print(f"    - ID: {HF_REPO_ID}")
    
    except Exception as e:
        print(f"  ✗ Cannot access repository: {e}")
        print(f"    Make sure you have write permissions")
        return False
    
    return True, token

def upload_batch(token: str):
    """
    Upload entire models_v6 folder in batch mode
    """
    print_section("Batch Upload")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        
        print(f"Source: {LOCAL_MODELS_DIR}")
        print(f"Target: {HF_REPO_ID}/{REMOTE_DIR}")
        print(f"Mode: Batch upload (one commit)")
        print()
        
        # Get file count for display
        files = list(Path(LOCAL_MODELS_DIR).glob('*'))
        print(f"Preparing {len(files)} files...")
        for f in sorted(files)[:10]:  # Show first 10
            print(f"  • {f.name} ({format_size(f.stat().st_size)})")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        print()
        
        # Attempt batch upload with retries
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[上傳中] Batch uploading all files...", end=' ', flush=True)
                
                api.upload_folder(
                    folder_path=LOCAL_MODELS_DIR,
                    repo_id=HF_REPO_ID,
                    repo_type=REPO_TYPE,
                    path_in_repo=REMOTE_DIR,
                    token=token,
                    commit_message=COMMIT_MESSAGE,
                    multi_commit=False,  # KEY: Single commit for batch
                    multi_commit_skip_errors=False
                )
                
                print(f"✓ Upload successful!\n")
                return True
            
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (attempt + 1)
                    print(f"\nERROR (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"\n✗ Upload failed after {MAX_RETRIES} attempts")
                    print(f"Error: {e}")
                    return False
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def print_summary(success: bool):
    """
    Print final summary and next steps
    """
    print_section("Summary")
    
    if success:
        print("✓ Upload Completed Successfully!\n")
        
        # Count files
        files = list(Path(LOCAL_MODELS_DIR).glob('*'))
        keras_files = [f for f in files if f.suffix == '.keras']
        json_files = [f for f in files if f.suffix == '.json']
        
        print(f"Files uploaded:")
        print(f"  .keras models: {len(keras_files)}")
        print(f"  .json metrics: {len(json_files)}")
        print(f"  Total: {len(files)} files")
        print()
        print(f"Repository: https://huggingface.co/datasets/{HF_REPO_ID}")
        print(f"Models directory: https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{REMOTE_DIR}")
        print()
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        print("✗ Upload Failed\n")
        print("Troubleshooting:")
        print("  1. Verify HF_TOKEN is correct")
        print("  2. Check that models_v6 folder exists and has files")
        print("  3. Verify network connection")
        print("  4. Try again: python upload_models_v6_batch.py")
    
    print("\n" + "="*80 + "\n")
    return success

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header()
    
    # Step 1: Validate environment
    print("[Step 1] Environment Validation")
    validation_result = validate_environment()
    
    if validation_result is False:
        print_summary(False)
        sys.exit(1)
    
    if isinstance(validation_result, tuple):
        success, token = validation_result
        if not success:
            print_summary(False)
            sys.exit(1)
    else:
        token = validation_result
    
    # Step 2: Upload batch
    print("\n[Step 2] Batch Upload")
    success = upload_batch(token)
    
    # Step 3: Summary
    print("[Step 3] Summary")
    print_summary(success)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
