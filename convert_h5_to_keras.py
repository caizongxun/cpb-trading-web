#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert .h5 models to .keras format
One-time conversion script to upgrade old models
"""

import os
from pathlib import Path
from tensorflow.keras.models import load_model

def convert_models():
    models_dir = Path("models_v6")
    
    if not models_dir.exists():
        print("✗ models_v6 directory not found!")
        return
    
    h5_files = list(models_dir.glob("*.h5"))
    
    if not h5_files:
        print("✗ No .h5 files found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Converting {len(h5_files)} .h5 models to .keras format")
    print(f"{'='*80}\n")
    
    success_count = 0
    failed_count = 0
    
    for h5_file in sorted(h5_files):
        keras_file = h5_file.with_suffix(".keras")
        
        # Skip if already converted
        if keras_file.exists():
            print(f"⊘ {h5_file.name} → Already converted, skipping")
            continue
        
        try:
            print(f"[轉換中] {h5_file.name}...", end=' ', flush=True)
            
            # Load .h5 model
            model = load_model(str(h5_file))
            
            # Save as .keras
            model.save(str(keras_file), save_format='keras')
            
            # Verify
            if keras_file.exists():
                h5_size = h5_file.stat().st_size / (1024*1024)
                keras_size = keras_file.stat().st_size / (1024*1024)
                print(f"✓ OK ({h5_size:.1f} MB → {keras_size:.1f} MB)")
                success_count += 1
            else:
                print(f"✗ FAILED")
                failed_count += 1
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Conversion Summary")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(h5_files)}")
    print(f"Failed: {failed_count}/{len(h5_files)}")
    
    if success_count > 0:
        # Check final state
        keras_files = list(models_dir.glob("*.keras"))
        json_files = list(models_dir.glob("*.json"))
        print(f"\nCurrent state:")
        print(f"  .keras files: {len(keras_files)}")
        print(f"  .json files: {len(json_files)}")
        print(f"  Total: {len(keras_files) + len(json_files)} files")
        
        if len(keras_files) == 60 and len(json_files) == 60:
            print(f"\n✓ Ready for upload! Execute: python upload_models_v6_batch.py")
        else:
            print(f"\n⚠ Waiting for more conversions...")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    convert_models()
