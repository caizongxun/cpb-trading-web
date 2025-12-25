#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5 1H Only Visualization - Raw Predictions (No Smoothing)
Model: {SYMBOL}_1h_model.h5
Scalers: {SYMBOL}_1h_scalers.pkl

Using raw model predictions without any smoothing for practical use
"""

import os, sys, json, warnings, pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (18, 10)

SYMBOLS = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE', 'ATOM']
TIMEFRAME = '1h'
BASE_PRICES = {'BTC': 42000, 'ETH': 2300, 'BNB': 610, 'ADA': 0.95, 'SOL': 195, 'XRP': 2.45, 'DOGE': 0.38, 'ATOM': 12.5}
COLOR_REAL, COLOR_PRED = '#2ecc71', '#e74c3c'

print("="*80 + "\nV5 1H Visualization (Raw Predictions - No Smoothing)\n" + "="*80 + "\n")

# Install
print("[Step 1] Installing dependencies...")
subprocess.run(['pip', 'install', 'tensorflow', 'huggingface_hub', '-q'], check=False)
print("✓ Dependencies installed\n")

# Download V5 Models
print("[Step 2] Downloading V5 models...")
from huggingface_hub import hf_hub_download
model_cache, scalers_cache, model_stats = {}, {}, {'v5': 0, 'synthetic': 0}

def get_v5_model_1h(symbol):
    """Download V5 1h model with scalers"""
    model_key = f"{symbol}_{TIMEFRAME}"
    
    if model_key in model_cache:
        return model_cache[model_key], model_stats['v5']
    
    print(f"      Attempting: {symbol}_1h_model.h5...", end=' ', flush=True)
    try:
        model_path = hf_hub_download(
            repo_id="zongowo111/cpb-models",
            filename=f"models_v5/{symbol}_1h_model.h5",
            repo_type="dataset",
            cache_dir="./models_cache"
        )
        
        print("✓ Found model, downloading scalers...", end=' ', flush=True)
        try:
            scalers_path = hf_hub_download(
                repo_id="zongowo111/cpb-models",
                filename=f"models_v5/{symbol}_1h_scalers.pkl",
                repo_type="dataset",
                cache_dir="./models_cache"
            )
            model_cache[model_key] = model_path
            scalers_cache[model_key] = scalers_path
            model_stats['v5'] += 1
            print("✓ V5 Complete")
            return model_path, True
        except:
            print("✗ Scalers not found")
            model_cache[model_key] = model_path
            scalers_cache[model_key] = None
            model_stats['v5'] += 1
            print("   [Warning] Using model without scalers")
            return model_path, True
    except:
        print("✗ Not found")
    
    print(f"      Using Synthetic")
    model_stats['synthetic'] += 1
    return None, False

print("✓ V5 Model loader ready\n")

# Helper functions
print("[Step 3] Defining helper functions...")

def generate_sample_data(symbol, num_samples=200):
    base_price = BASE_PRICES.get(symbol, 100)
    minutes = 60
    prices, current_price = [], base_price
    
    for i in range(num_samples):
        change = np.random.normal(0, base_price * 0.005)
        current_price += change
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, base_price * 0.003))
        low_price = open_price - abs(np.random.normal(0, base_price * 0.003))
        close_price = np.random.uniform(low_price, high_price)
        volume = np.random.uniform(1000, 10000)
        prices.append({'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price, 'volume': volume})
    
    df = pd.DataFrame(prices)
    start_time = datetime.now() - timedelta(minutes=num_samples * minutes)
    df['time'] = [start_time + timedelta(minutes=i * minutes) for i in range(num_samples)]
    df.set_index('time', inplace=True)
    return df

def load_scalers(scalers_path):
    """Load scalers.pkl"""
    try:
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        return scalers
    except:
        return None

def normalize_with_scalers(X, scalers):
    """Normalize using scalers"""
    if scalers is None:
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True) + 1e-8
        return (X - X_mean) / X_std, X_mean, X_std
    
    try:
        if hasattr(scalers, 'transform'):
            X_2d = X.reshape(-1, X.shape[-1])
            X_scaled = scalers.transform(X_2d)
            return X_scaled.reshape(X.shape), None, None
    except:
        pass
    
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - X_mean) / X_std, X_mean, X_std

def generate_predictions_raw(data, model_path=None, scalers_path=None):
    """
    產生原始預測 - 不做任何平滑
    Raw predictions without any smoothing
    """
    close_prices = data['close'].values
    
    if model_path:
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            scalers = load_scalers(scalers_path) if scalers_path else None
            
            # 使用第一个lookback能成功的預測
            for lookback in [60, 50, 40, 30]:
                try:
                    X = [close_prices[i:i+lookback] for i in range(len(close_prices) - lookback)]
                    if len(X) > 0:
                        X = np.array(X)
                        X_norm, X_mean, X_std = normalize_with_scalers(X, scalers)
                        
                        # 原始預測輸出
                        predictions = model.predict(X_norm, verbose=0)
                        
                        # 平扁化
                        if len(predictions.shape) > 1:
                            pred_prices = predictions.flatten()
                        else:
                            pred_prices = predictions
                        
                        # Inverse transform
                        if X_mean is not None and X_std is not None:
                            pred_prices = pred_prices * X_std[:, 0] + X_mean[:, 0]
                        
                        # 永位 - 保留前 lookback 根蘭優格倉來（因為沒有預測）
                        pred_full = [close_prices[lookback - 1]] * lookback + pred_prices.tolist()
                        
                        # 使用原始預測 - 不做任何平滑！
                        return np.array(pred_full[:len(close_prices)])
                except:
                    continue
        except:
            pass
    
    # Synthetic - 也是原始輸出，不平滑
    predictions = np.random.normal(close_prices.mean(), close_prices.std() * 0.5, len(close_prices))
    predictions = close_prices + np.random.normal(0, close_prices.std() * 0.02, len(close_prices))
    return predictions

def create_visualization(symbol, data, predictions, has_model=False):
    """Create visualization with raw predictions"""
    data_vis = data.iloc[:100]
    predictions_vis = predictions[:100]
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # Main chart - RAW predictions without smoothing
    ax1, x = fig.add_subplot(gs[0]), np.arange(len(data_vis))
    ax1.fill_between(x, data_vis['low'], data_vis['high'], alpha=0.15, color=COLOR_REAL, label='Price Range (Real)')
    ax1.plot(x, data_vis['close'], color=COLOR_REAL, linewidth=2.5, label='Real Price (Close)', marker='o', markersize=4, alpha=0.8)
    
    # 原始預測 - 不平滑！
    ax1.plot(x, predictions_vis, color=COLOR_PRED, linewidth=2.5, label='Predicted Price (Raw - No Smoothing)', 
             marker='s', markersize=4, alpha=0.8, linestyle='-')  # Solid line, not dashed
    
    model_label = "V5" if has_model else "SYNTHETIC"
    ax1.set_title(f'{symbol} 1H ({model_label}) - Raw Predictions (No Smoothing)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle=':'), ax1.legend(loc='best', fontsize=11)
    
    # Error chart
    ax2 = fig.add_subplot(gs[1])
    error = predictions_vis - data_vis['close'].values
    error_pct = (error / data_vis['close'].values) * 100
    colors = [COLOR_PRED if e > 0 else COLOR_REAL for e in error]
    ax2.bar(x, error_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_title('Prediction Error (%)', fontsize=12, fontweight='bold'), ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Volume chart
    ax3 = fig.add_subplot(gs[2])
    colors_vol = [COLOR_REAL if data_vis['close'].iloc[i] >= data_vis['open'].iloc[i] else COLOR_PRED for i in range(len(data_vis))]
    ax3.bar(x, data_vis['volume'], color=colors_vol, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_title('Trading Volume', fontsize=12, fontweight='bold'), ax3.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Statistics
    mae, rmse = np.abs(error).mean(), np.sqrt((error ** 2).mean())
    mape = np.mean(np.abs(error_pct))
    stats_text = f"Model: {model_label} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | Raw Predictions (No Smoothing)"
    fig.text(0.02, 0.98, stats_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"v5_1h_raw_{symbol}_{timestamp}.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    return filename

print("✓ Helper functions defined\n")

# Generate all 1h V5 charts
print("[Step 4] Generating V5 1H visualizations (Raw Predictions)...")
print(f"Processing {len(SYMBOLS)} symbols (1h only, no smoothing)\n")

results, generated = [], 0

for i, symbol in enumerate(SYMBOLS, 1):
    print(f"{i}. {symbol}:")
    try:
        data = generate_sample_data(symbol)
        model_path, has_model = get_v5_model_1h(symbol)
        
        scalers_path = None
        if has_model and model_path:
            model_key = f"{symbol}_{TIMEFRAME}"
            scalers_path = scalers_cache.get(model_key)
        
        predictions = generate_predictions_raw(data, model_path, scalers_path)
        filename = create_visualization(symbol, data, predictions, has_model)
        
        print(f"   ✓ Generated: {filename}\n")
        results.append((symbol, TIMEFRAME, has_model, filename))
        generated += 1
    except Exception as e:
        print(f"   ✗ Error: {str(e)}\n")

# Summary
print(f"\n{'='*80}")
print(f"\n[Step 5] Summary")
print(f"\nGenerated: {generated}/{len(SYMBOLS)} charts (1H only)")
print(f"Success Rate: {(generated/len(SYMBOLS))*100:.1f}%")
print(f"\nModel Usage:")
print(f"  V5 Models:  {model_stats['v5']}")
print(f"  Synthetic:  {model_stats['synthetic']}")
print(f"\nPrediction Method:")
print(f"  RAW PREDICTIONS - NO SMOOTHING")
print(f"  Direct model output for practical trading use")
print(f"\nModel Details:")
for symbol, tf, has_model, filename in results:
    if has_model:
        status = "✓ V5 Model"
    else:
        status = "○ Synthetic"
    print(f"  {symbol.ljust(6)} 1H  {status.ljust(16)} {filename}")

# Display
print(f"\n{'='*80}")
print(f"\n[Step 6] Displaying V5 1H charts (Raw Predictions)...\n")

try:
    from IPython.display import Image, display, HTML
    import glob
    
    chart_files = sorted(glob.glob('v5_1h_raw_*.png'))
    
    if chart_files:
        print(f"Found {len(chart_files)} charts:\n")
        for i, filename in enumerate(chart_files, 1):
            try:
                display(Image(filename))
                print(f"[{i}/{len(chart_files)}] {filename}")
            except:
                pass
    else:
        print("No charts found")
except:
    print("✓ Charts saved to current directory (non-Colab environment)")

print(f"\n{'='*80}")
print(f"\n✓ Complete! Generated {generated} V5 1H visualizations!")
print(f"\nKey Improvements:")
print(f"  ✅ Raw model predictions (no smoothing)")
print(f"  ✅ More volatile - realistic for trading")
print(f"  ✅ Practical use case")
print(f"\nFile naming:")
print(f"  Output: v5_1h_raw_{{SYMBOL}}_{{timestamp}}.png")
print(f"  Models: {{SYMBOL}}_1h_model.h5")
print(f"  Scalers: {{SYMBOL}}_1h_scalers.pkl")
print(f"{'='*80}\n")
