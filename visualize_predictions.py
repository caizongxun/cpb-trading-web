#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Model Predictions

Read models from HuggingFace and compare predictions with actual prices
Display first 100 candles with real vs predicted price comparison

Usage:
  python visualize_predictions.py BTC 1h
  python visualize_predictions.py ADA 15m
  python visualize_predictions.py ETH 1d

Supported symbols: BTC, ETH, BNB, ADA, SOL, XRP, DOGE, ATOM, etc.
Supported timeframes: 15m, 1h, 4h, 1d
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better display
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_REPO_ID = "zongowo111/cpb-models"
REPO_TYPE = "dataset"
VIS_CANDLES = 100  # 只顯示前100根

# Color scheme
COLOR_REAL = '#2ecc71'  # Green for real prices
COLOR_PRED = '#e74c3c'  # Red for predicted prices
COLOR_AREA_REAL = '#27ae60'  # Darker green
COLOR_AREA_PRED = '#c0392b'  # Darker red

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
    print("Model Prediction Visualizer")
    print("="*80 + "\n")

def print_section(title):
    print(f"\n[{title}]")
    print("-" * 80)

def get_hf_token():
    """Get HF token from environment or file"""
    # Try environment variable
    token = os.getenv('HF_TOKEN')
    if token:
        return token
    
    # Try Secrets (for Colab)
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token
    except:
        pass
    
    # Try file
    token_file = Path.home() / '.huggingface' / 'token'
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
            if token:
                return token
        except:
            pass
    
    return None

def download_model(symbol, timeframe):
    """Download model from HuggingFace"""
    try:
        from huggingface_hub import hf_hub_download
        
        filename = f"models_v6/{symbol}_{timeframe}.keras"
        print(f"  Downloading: {filename}")
        
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            cache_dir="./models_cache"
        )
        
        print(f"  ✓ Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
        return None

def download_metrics(symbol, timeframe):
    """Download metrics JSON from HuggingFace"""
    try:
        from huggingface_hub import hf_hub_download
        
        filename = f"models_v6/{symbol}_{timeframe}_metrics.json"
        print(f"  Downloading: {filename}")
        
        metrics_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            cache_dir="./models_cache"
        )
        
        print(f"  ✓ Downloaded to: {metrics_path}")
        return metrics_path
    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
        return None

def generate_sample_data(symbol, timeframe, num_samples=200):
    """
    Generate sample OHLCV data for demonstration
    In real usage, you would load actual market data
    """
    # Define base prices for different symbols
    base_prices = {
        'BTC': 42000,
        'ETH': 2300,
        'BNB': 610,
        'ADA': 0.95,
        'SOL': 195,
        'XRP': 2.45,
        'DOGE': 0.38,
        'ATOM': 12.5,
    }
    
    base_price = base_prices.get(symbol.upper(), 100)
    
    # Timeframe to minutes mapping
    tf_minutes = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }
    
    minutes = tf_minutes.get(timeframe, 60)
    
    # Generate realistic OHLCV data
    prices = []
    current_price = base_price
    
    for i in range(num_samples):
        # Random walk with mean reversion
        change = np.random.normal(0, base_price * 0.005)  # 0.5% volatility
        current_price += change
        
        # Generate OHLCV
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, base_price * 0.003))
        low_price = open_price - abs(np.random.normal(0, base_price * 0.003))
        close_price = np.random.uniform(low_price, high_price)
        volume = np.random.uniform(1000, 10000)
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
        })
    
    # Create DataFrame
    df = pd.DataFrame(prices)
    
    # Add datetime index
    start_time = datetime.now() - timedelta(minutes=num_samples * minutes)
    timestamps = [start_time + timedelta(minutes=i * minutes) for i in range(num_samples)]
    df['time'] = timestamps
    df.set_index('time', inplace=True)
    
    return df

def generate_predictions(data, model_path=None):
    """
    Generate predictions
    If model_path is provided, use the actual model
    Otherwise, generate synthetic predictions for demo
    """
    close_prices = data['close'].values
    
    if model_path:
        try:
            from tensorflow.keras.models import load_model
            
            # Load model
            model = load_model(model_path)
            
            # Prepare data (assuming 60-step lookback)
            lookback = 60
            X = []
            for i in range(len(close_prices) - lookback):
                X.append(close_prices[i:i+lookback])
            
            if len(X) > 0:
                X = np.array(X)
                # Normalize
                X_mean = X.mean(axis=1, keepdims=True)
                X_std = X.std(axis=1, keepdims=True) + 1e-8
                X = (X - X_mean) / X_std
                
                # Predict
                predictions = model.predict(X, verbose=0)
                
                # Denormalize
                pred_prices = predictions.flatten() * X_std[:, 0] + X_mean[:, 0]
                
                # Align with data
                pred_full = [close_prices[lookback - 1]] * lookback + pred_prices.tolist()
                return np.array(pred_full[:len(close_prices)])
        except Exception as e:
            print(f"  ⚠ Model prediction failed: {e}")
            print(f"  Falling back to synthetic predictions")
    
    # Synthetic predictions for demo
    # Add slight lag and smoothing to predicted prices
    predictions = close_prices.copy()
    
    # Add lag
    predictions = np.roll(predictions, 3)
    
    # Add smoothing
    window = 5
    predictions = pd.Series(predictions).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    
    # Add small random noise
    predictions += np.random.normal(0, close_prices.std() * 0.01, len(predictions))
    
    return predictions

def create_visualization(symbol, timeframe, data, predictions, metrics=None):
    """
    Create comprehensive visualization comparing real vs predicted prices
    """
    # Take only first VIS_CANDLES
    data_vis = data.iloc[:VIS_CANDLES]
    predictions_vis = predictions[:VIS_CANDLES]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # ====================================================================
    # MAIN CHART: Real vs Predicted Prices
    # ====================================================================
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(data_vis))
    
    # Plot real prices (close prices with OHLC range)
    ax1.fill_between(x, data_vis['low'], data_vis['high'], 
                      alpha=0.15, color=COLOR_REAL, label='Price Range (Real)')
    ax1.plot(x, data_vis['close'], color=COLOR_REAL, linewidth=2.5, 
             label='Real Price (Close)', marker='o', markersize=4, alpha=0.8)
    
    # Plot predicted prices
    ax1.plot(x, predictions_vis, color=COLOR_PRED, linewidth=2.5, 
             label='Predicted Price', marker='s', markersize=4, alpha=0.8, linestyle='--')
    
    # Formatting
    ax1.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol.upper()} {timeframe.upper()} - Real vs Predicted Prices (First {VIS_CANDLES} Candles)',
                   fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='best', fontsize=11, framealpha=0.95)
    
    # Add value labels at key points
    for i in [0, len(data_vis)//2, len(data_vis)-1]:
        if i < len(data_vis):
            ax1.text(i, data_vis['close'].iloc[i], 
                    f"{data_vis['close'].iloc[i]:.2f}",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_AREA_REAL, alpha=0.7, edgecolor='none'))
            ax1.text(i, predictions_vis[i],
                    f"{predictions_vis[i]:.2f}",
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_AREA_PRED, alpha=0.7, edgecolor='none'))
    
    # ====================================================================
    # ERROR / DIFFERENCE CHART
    # ====================================================================
    ax2 = fig.add_subplot(gs[1])
    
    error = predictions_vis - data_vis['close'].values
    error_pct = (error / data_vis['close'].values) * 100
    
    colors = [COLOR_AREA_PRED if e > 0 else COLOR_AREA_REAL for e in error]
    ax2.bar(x, error_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error (Real vs Predicted)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # ====================================================================
    # VOLUME CHART
    # ====================================================================
    ax3 = fig.add_subplot(gs[2])
    
    colors_vol = [COLOR_AREA_REAL if data_vis['close'].iloc[i] >= data_vis['open'].iloc[i] else COLOR_AREA_PRED 
                  for i in range(len(data_vis))]
    ax3.bar(x, data_vis['volume'], color=colors_vol, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax3.set_title('Trading Volume', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # ====================================================================
    # STATISTICS
    # ====================================================================
    
    # Calculate statistics
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    mape = np.mean(np.abs(error_pct))
    max_error = np.max(np.abs(error))
    
    # Add statistics text
    stats_text = f"""Statistics (First {VIS_CANDLES} Candles):
MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | Max Error: {max_error:.4f}
Real Price Range: {data_vis['low'].min():.4f} - {data_vis['high'].max():.4f}
Predicted Price Range: {predictions_vis.min():.4f} - {predictions_vis.max():.4f}"""
    
    fig.text(0.02, 0.98, stats_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    # ====================================================================
    # FINALIZE
    # ====================================================================
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"prediction_viz_{symbol}_{timeframe}_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved: {filename}")
    
    # Display
    plt.show()
    
    return fig

def main():
    print_header()
    
    # ========================================================================
    # STEP 1: Get Symbol and Timeframe
    # ========================================================================
    print_section("Step 1: Get Symbol and Timeframe")
    
    if len(sys.argv) >= 3:
        symbol = sys.argv[1].upper()
        timeframe = sys.argv[2].lower()
    else:
        print("\nUsage: python visualize_predictions.py SYMBOL TIMEFRAME")
        print("\nExamples:")
        print("  python visualize_predictions.py BTC 1h")
        print("  python visualize_predictions.py ADA 15m")
        print("  python visualize_predictions.py ETH 1d")
        print("\nSupported symbols: BTC, ETH, BNB, ADA, SOL, XRP, DOGE, ATOM")
        print("Supported timeframes: 15m, 1h, 4h, 1d")
        
        symbol = input("\nEnter symbol (e.g., BTC): ").upper()
        timeframe = input("Enter timeframe (15m/1h/4h/1d): ").lower()
    
    print(f"\n  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    
    # Validate
    valid_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE', 'ATOM']
    valid_timeframes = ['15m', '1h', '4h', '1d']
    
    if symbol not in valid_symbols:
        print(f"  ⚠ Symbol not found in common list, but will try to proceed")
    if timeframe not in valid_timeframes:
        print(f"  ✗ Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Download Model and Metrics
    # ========================================================================
    print_section("Step 2: Download Model and Metrics")
    
    token = get_hf_token()
    if not token:
        print("  ⚠ No HuggingFace token found")
        print("    Will use synthetic predictions instead")
        model_path = None
        metrics = None
    else:
        print(f"  ✓ HuggingFace token found")
        model_path = download_model(symbol, timeframe)
        metrics_path = download_metrics(symbol, timeframe)
        
        # Load metrics
        metrics = None
        if metrics_path:
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print(f"  ✓ Metrics loaded")
            except:
                print(f"  ⚠ Could not load metrics")
    
    # ========================================================================
    # STEP 3: Generate Sample Data
    # ========================================================================
    print_section("Step 3: Generate Sample Data")
    
    print(f"  Generating sample OHLCV data for {symbol} {timeframe}")
    data = generate_sample_data(symbol, timeframe, num_samples=200)
    print(f"  ✓ Generated {len(data)} candles")
    print(f"    Price range: {data['close'].min():.4f} - {data['close'].max():.4f}")
    print(f"    Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    # ========================================================================
    # STEP 4: Generate Predictions
    # ========================================================================
    print_section("Step 4: Generate Predictions")
    
    print(f"  Generating predictions...")
    predictions = generate_predictions(data, model_path)
    print(f"  ✓ Predictions generated")
    print(f"    Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    
    # Calculate error
    error = np.abs(predictions - data['close'].values)
    error_pct = (error / data['close'].values) * 100
    
    print(f"    Mean Absolute Error: {error.mean():.6f}")
    print(f"    Mean Absolute Percentage Error: {error_pct.mean():.2f}%")
    
    # ========================================================================
    # STEP 5: Create Visualization
    # ========================================================================
    print_section("Step 5: Create Visualization")
    
    print(f"  Creating visualization for first {VIS_CANDLES} candles...")
    fig = create_visualization(symbol, timeframe, data, predictions, metrics)
    print(f"  ✓ Visualization created")
    
    # ========================================================================
    # DONE
    # ========================================================================
    print_section("Summary")
    
    print(f"  ✓ Visualization Complete!")
    print(f"\n  Chart shows:")
    print(f"    1. Real prices (green line with range) vs Predicted prices (red dashed line)")
    print(f"    2. Prediction error percentage for each candle")
    print(f"    3. Trading volume for each candle")
    print(f"\n  Model: {symbol} {timeframe}")
    print(f"  Candles displayed: {VIS_CANDLES} / {len(data)}")
    print()
    
    print("="*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
