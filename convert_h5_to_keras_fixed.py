#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert .h5 models to .keras format (FIXED VERSION)
Handles custom loss functions and version incompatibilities
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def convert_models_fixed():
    models_dir = Path("models_v6")
    
    if not models_dir.exists():
        print("✗ models_v6 directory not found!")
        return
    
    h5_files = sorted(models_dir.glob("*.h5"))
    
    if not h5_files:
        print("✗ No .h5 files found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Converting {len(h5_files)} .h5 models to .keras format (FIXED)")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"{'='*80}\n")
    
    success_count = 0
    failed_count = 0
    failed_models = []
    
    for h5_file in h5_files:
        keras_file = h5_file.with_suffix(".keras")
        
        # Skip if already converted
        if keras_file.exists():
            print(f"⊘ {h5_file.name:25} → Already converted, skipping")
            success_count += 1
            continue
        
        try:
            print(f"[轉換中] {h5_file.name:25}", end=' ', flush=True)
            
            # Method 1: Load with custom objects (handles MSE loss)
            # 'mse' is actually just an alias for MeanSquaredError
            custom_objects = {
                'mse': MeanSquaredError(),
                'MSE': MeanSquaredError(),
            }
            
            # Load .h5 model with custom objects
            model = load_model(
                str(h5_file),
                custom_objects=custom_objects,
                safe_mode=False  # Required for older model compatibility
            )
            
            # Save as .keras format
            # This will use the new SavedModel format which is more robust
            model.save(str(keras_file), save_format='keras')
            
            # Verify conversion was successful
            if keras_file.exists():
                h5_size = h5_file.stat().st_size / (1024*1024)
                keras_size = keras_file.stat().st_size / (1024*1024)
                print(f"✓ SUCCESS ({h5_size:.1f} MB → {keras_size:.1f} MB)")
                success_count += 1
            else:
                print(f"✗ FAILED (file not created)")
                failed_count += 1
                failed_models.append(h5_file.name)
        
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:60]}")
            failed_count += 1
            failed_models.append(h5_file.name)
            
            # Try alternative method: rebuild model
            if "mse" in str(e).lower():
                print(f"     → Attempting alternative method...")
                try:
                    # Load with safe mode disabled
                    model = load_model(str(h5_file), safe_mode=False)
                    model.save(str(keras_file), save_format='keras')
                    
                    if keras_file.exists():
                        print(f"     ✓ Alternative method SUCCESS")
                        success_count += 1
                        failed_count -= 1
                        failed_models.pop()
                except Exception as e2:
                    print(f"     ✗ Alternative method also failed: {str(e2)[:40]}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Conversion Summary")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(h5_files)}")
    print(f"Failed: {failed_count}/{len(h5_files)}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model_name in failed_models:
            print(f"  - {model_name}")
    
    if success_count > 0:
        # Check final state
        keras_files = list(models_dir.glob("*.keras"))
        json_files = list(models_dir.glob("*.json"))
        print(f"\nCurrent state:")
        print(f"  .keras files: {len(keras_files)}")
        print(f"  .json files: {len(json_files)}")
        print(f"  .h5 files: {len(list(models_dir.glob('*.h5')))}")
        print(f"  Total converted: {len(keras_files) + len(json_files)}")
        
        if len(keras_files) + len(json_files) >= len(h5_files):
            print(f"\n✓ Conversion complete! Ready for upload.")
            print(f"  Execute: python upload_models_v6_batch.py")
        else:
            remaining = len(h5_files) - (len(keras_files) + len(json_files))
            print(f"\n⚠ {remaining} models still need conversion")
    
    print(f"\n{'='*80}\n")
    
    return success_count, failed_count

if __name__ == "__main__":
    success, failed = convert_models_fixed()
    sys.exit(0 if failed == 0 else 1)
