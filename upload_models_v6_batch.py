#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Upload V6 Models to HuggingFace (FIXED VERSION)

Uploads entire models_v6 directory to HF in ONE batch operation
Avoids API rate limits and ensures complete data consistency

Usage:
  In Colab:
    !pip install huggingface_hub --upgrade
    !python upload_models_v6_batch.py
  
  Locally:
    pip install huggingface_hub --upgrade
    export HF_TOKEN='your_token_here'
    python upload_models_v6_batch.py

Model Format: .keras (modern TensorFlow format)
"""

import os
import sys
import time
import getpass
from pathlib import Path
from datetime import datetime
import json

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
    print("HuggingFace Batch Uploader - models_v6 Directory (FIXED)")
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
    h5_files = [f for f in files if f.suffix == '.h5']
    total_size = sum(f.stat().st_size for f in files)
    
    print(f"  ✓ Directory exists, contains {len(files)} files")
    print(f"    - .keras files: {len(keras_files)}")
    print(f"    - .json files: {len(json_files)}")
    print(f"    - .h5 files: {len(h5_files)} (legacy)")
    print(f"    - Total size: {format_size(total_size)}")
    
    if len(keras_files) == 0:
        print(f"  ⚠ Warning: No .keras files found. Make sure conversion completed.")
    
    # Check HF Token
    print(f"\nHuggingFace Token")
    token, source = get_token_from_sources()
    
    if not token:
        print(f"  ⚠ No automated token found (Env/Secrets/File)")
        print(f"  ➔ Please enter your HuggingFace Write Token below:")
        try:
            token = getpass.getpass("  HF Token > ")
            if token and token.strip():
                token = token.strip()
                source = "manual input"
                print(f"  ✓ Token received")
            else:
                print(f"  ✗ No token provided!")
                return False
        except Exception as e:
            print(f"  ✗ Error reading input: {e}")
            return False
    else:
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
    Uses CommitScheduler for reliable batch uploads
    """
    print_section("Batch Upload")
    
    try:
        from huggingface_hub import HfApi, CommitScheduler
        
        api = HfApi(token=token)
        
        print(f"Source: {LOCAL_MODELS_DIR}")
        print(f"Target: {HF_REPO_ID}/{REMOTE_DIR}")
        print(f"Method: CommitScheduler (reliable batch upload)")
        print()
        
        # Get file list
        files = sorted(Path(LOCAL_MODELS_DIR).glob('*'))
        print(f"Preparing {len(files)} files...")
        for f in files[:10]:  # Show first 10
            print(f"  • {f.name} ({format_size(f.stat().st_size)})")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        print()
        
        # Upload using CommitScheduler (most reliable method)
        print(f"[上傳中] Starting batch upload...", flush=True)
        
        # Use CommitScheduler for batch upload
        scheduler = CommitScheduler(
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_every=float('inf'),  # Upload everything in one commit
        )
        
        with scheduler:
            # Upload each file
            for file_path in files:
                print(f"  → {file_path.name}...", end=' ', flush=True)
                try:
                    scheduler.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=f"{REMOTE_DIR}/{file_path.name}",
                    )
                    print("✓")
                except Exception as e:
                    print(f"✗ ({e})")
        
        print(f"\n[上傳中] Finalizing batch commit...")
        print(f"✓ Upload successful!\n")
        return True
    
    except ImportError:
        # Fallback to direct upload method if CommitScheduler not available
        print(f"[上傳中] Using direct upload method (CommitScheduler not available)...\n")
        return upload_batch_direct(token)
    
    except Exception as e:
        print(f"\n✗ Upload error: {e}")
        print(f"\n[上傳中] Retrying with direct upload method...\n")
        return upload_batch_direct(token)

def upload_batch_direct(token: str):
    """
    Fallback: Direct upload method compatible with older huggingface_hub versions
    Uploads files one by one but in a single commit operation
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        files = sorted(Path(LOCAL_MODELS_DIR).glob('*'))
        
        print(f"Uploading {len(files)} files in direct mode...")
        
        for attempt in range(MAX_RETRIES):
            try:
                # Collect all file operations
                print(f"\n[上傳中] Batch {attempt + 1}/{MAX_RETRIES}...")
                
                # Upload each file individually
                for idx, file_path in enumerate(files, 1):
                    print(f"  [{idx}/{len(files)}] {file_path.name}...", end=' ', flush=True)
                    
                    try:
                        with open(file_path, 'rb') as f:
                            api.upload_file(
                                path_or_fileobj=f,
                                path_in_repo=f"{REMOTE_DIR}/{file_path.name}",
                                repo_id=HF_REPO_ID,
                                repo_type=REPO_TYPE,
                                token=token,
                                commit_message=COMMIT_MESSAGE if idx == len(files) else None,
                            )
                        print("✓")
                    except Exception as e:
                        print(f"✗ ({str(e)[:40]})")
                        # Continue with next file
                        continue
                
                print(f"\n✓ Upload completed!\n")
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
        print(f"✗ Unexpected error: {e}")
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
        h5_files = [f for f in files if f.suffix == '.h5']
        
        print(f"Files uploaded:")
        print(f"  .keras models: {len(keras_files)}")
        print(f"  .json metrics: {len(json_files)}")
        print(f"  .h5 legacy: {len(h5_files)}")
        print(f"  Total: {len(files)} files")
        print()
        print(f"Repository: https://huggingface.co/datasets/{HF_REPO_ID}")
        print(f"Models dir: https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{REMOTE_DIR}")
        print()
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        print("✗ Upload Failed\n")
        print("Troubleshooting:")
        print("  1. Verify HF_TOKEN is correct and has write permissions")
        print("  2. Check that models_v6 folder exists and has files")
        print("  3. Verify network connection")
        print("  4. Try updating huggingface_hub: pip install --upgrade huggingface_hub")
        print("  5. Try again: python upload_models_v6_batch.py")
    
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
