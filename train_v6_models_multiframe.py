#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading V6 Multi-Frame Multi-Coin Model Training
Train LSTM models for multi-step forecasting (30 bars input -> 10 bars output)

Supported:
- Coins: 20+ major cryptocurrencies
- Timeframes: 1d (daily), 1h (hourly), 15m (15-min)
- Output: OHLC (Open, High, Low, Close) for next 10 bars
- Inference: Direct to Hugging Face models_v6 directory
- Format: Modern .keras (replaces deprecated .h5)

Usage in Colab:
    !pip install yfinance pandas numpy tensorflow scikit-learn huggingface-hub -q
    !git clone https://github.com/caizongxun/cpb-trading-web.git
    %cd cpb-trading-web
    !python train_v6_models_multiframe.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

COINS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BNB': 'BNB-USD',
    'SOL': 'SOL-USD',
    'XRP': 'XRP-USD',
    'ADA': 'ADA-USD',
    'DOGE': 'DOGE-USD',
    'AVAX': 'AVAX-USD',
    'LINK': 'LINK-USD',
    'DOT': 'DOT-USD',
    'LTC': 'LTC-USD',
    'ATOM': 'ATOM-USD',
    'UNI': 'UNI-USD',
    'MATIC': 'MATIC-USD',
    'NEAR': 'NEAR-USD',
    'FTM': 'FTM-USD',
    'CRO': 'CRO-USD',
    'VET': 'VET-USD',
    'ICP': 'ICP-USD',
    'HBAR': 'HBAR-USD',
}

TIMEFRAMES = {
    '1d': {'period': '2y', 'interval': '1d'},
    '1h': {'period': '60d', 'interval': '1h'},
    '15m': {'period': '14d', 'interval': '15m'},
}

MODEL_PARAMS = {
    'lookback': 30,      # Input sequence length
    'forecast': 10,      # Output sequence length (next 10 bars)
    'lstm_units': 128,
    'dropout': 0.2,
    'dense_units': 64,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping_patience': 15,
}

COLAB_MODE = True  # Set to False if running locally
HF_TOKEN = None    # Will be requested in Colab
HF_REPO_ID = "zongowo111/cpb-models"  # HF repository

print("\n" + "="*80)
print("CPB TRADING V6 - MULTI-FRAME MULTI-COIN MODEL TRAINER")
print("="*80)
print(f"Coins: {len(COINS)}")
print(f"Timeframes: {list(TIMEFRAMES.keys())}")
print(f"Total combinations: {len(COINS) * len(TIMEFRAMES)}")
print(f"Model Format: .keras (modern TensorFlow format)")
print("="*80 + "\n")

# ============================================================================
# DATA FETCHING
# ============================================================================

class DataFetcher:
    def __init__(self):
        self.cache = {}
    
    def fetch(self, symbol: str, period: str, interval: str, max_retries: int = 3):
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        for attempt in range(max_retries):
            try:
                print(f"  [Fetch] {symbol} {period}/{interval} (attempt {attempt+1}/{max_retries})...", end=' ')
                df = yf.download(symbol, period=period, interval=interval, progress=False)
                
                if df.empty:
                    print("EMPTY")
                    return None
                
                print(f"OK ({len(df)} bars)")
                self.cache[cache_key] = df
                return df
            
            except Exception as e:
                print(f"ERROR: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                continue
        
        print(f"  [Fetch] {symbol} FAILED after {max_retries} retries")
        return None

# ============================================================================
# SEQUENCE PREPARATION
# ============================================================================

class SequenceBuilder:
    def __init__(self, lookback: int, forecast: int):
        self.lookback = lookback
        self.forecast = forecast
        self.scalers = {}  # One scaler per feature
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: list = ['Open', 'High', 'Low', 'Close']):
        """
        Prepare sequences from OHLC data.
        
        Args:
            df: DataFrame with OHLC data
            feature_cols: Columns to use (default: ['Open', 'High', 'Low', 'Close'])
            
        Returns:
            X: (n_sequences, lookback, n_features)
            y: (n_sequences, forecast, n_features)
            scalers: dict of MinMaxScalers per feature
        """
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        data = df[available_cols].values.astype(np.float32)
        
        n_features = len(available_cols)
        min_rows = self.lookback + self.forecast
        
        if len(data) < min_rows:
            return None, None, None
        
        # Normalize each feature independently
        scalers = {}
        scaled_data = np.zeros_like(data)
        
        for i, col in enumerate(available_cols):
            scaler = MinMaxScaler()
            scaled_data[:, i] = scaler.fit_transform(data[:, i:i+1]).flatten()
            scalers[col] = scaler
        
        # Build sequences
        X, y = [], []
        for i in range(len(scaled_data) - min_rows + 1):
            X.append(scaled_data[i:i+self.lookback])  # (lookback, n_features)
            y.append(scaled_data[i+self.lookback:i+self.lookback+self.forecast])  # (forecast, n_features)
        
        self.scalers = scalers
        return np.array(X), np.array(y), scalers

# ============================================================================
# MODEL BUILDER
# ============================================================================

class ModelBuilder:
    @staticmethod
    def build(lookback: int, forecast: int, n_features: int, 
              lstm_units: int = 128, dropout: float = 0.2, dense_units: int = 64):
        """
        Build a Seq2Seq-style LSTM for multi-step forecasting.
        Input: (lookback, n_features) -> Output: (forecast, n_features)
        """
        model = Sequential([
            LSTM(lstm_units, activation='relu', input_shape=(lookback, n_features), return_sequences=True),
            Dropout(dropout),
            LSTM(lstm_units, activation='relu', return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation='relu'),
            Dense(forecast * n_features),  # Flatten output: forecast * n_features
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    @staticmethod
    def reshape_predictions(y_pred, forecast: int, n_features: int):
        """Reshape flattened predictions back to (batch, forecast, n_features)"""
        return y_pred.reshape(-1, forecast, n_features)

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(self, model_params: dict):
        self.params = model_params
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, n_features: int, verbose: int = 1):
        # Reshape y for training (flatten: batch, forecast * n_features)
        y_train_flat = y_train.reshape(y_train.shape[0], -1)
        y_val_flat = y_val.reshape(y_val.shape[0], -1)
        
        model = ModelBuilder.build(
            lookback=self.params['lookback'],
            forecast=self.params['forecast'],
            n_features=n_features,
            lstm_units=self.params['lstm_units'],
            dropout=self.params['dropout'],
            dense_units=self.params['dense_units'],
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.params['early_stopping_patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]
        
        self.history = model.fit(
            X_train, y_train_flat,
            validation_data=(X_val, y_val_flat),
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=callbacks,
            verbose=verbose,
        )
        
        return model

# ============================================================================
# EVALUATION & METRICS
# ============================================================================

class Evaluator:
    @staticmethod
    def evaluate(model, X_val, y_val, scalers: dict, feature_cols: list):
        y_val_flat = y_val.reshape(y_val.shape[0], -1)
        y_pred_flat = model.predict(X_val, verbose=0)
        y_pred = ModelBuilder.reshape_predictions(y_pred_flat, y_val.shape[1], y_val.shape[2])
        
        # Inverse scale for actual values
        metrics = {}
        for i, col in enumerate(feature_cols):
            if col not in scalers:
                continue
            
            scaler = scalers[col]
            y_true_col = scaler.inverse_transform(y_val[:, :, i])
            y_pred_col = scaler.inverse_transform(y_pred[:, :, i])
            
            mae = mean_absolute_error(y_true_col, y_pred_col)
            rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
            mape = mean_absolute_percentage_error(y_true_col, y_pred_col) * 100
            
            metrics[col] = {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}
        
        return metrics

# ============================================================================
# MODEL SAVING (.keras format)
# ============================================================================

class ModelSaver:
    @staticmethod
    def save_local(model, symbol: str, timeframe: str, metrics: dict, output_dir: str = 'models_v6'):
        """
        Save model in modern .keras format (replaces .h5)
        
        Args:
            model: Trained Keras model
            symbol: Coin symbol (e.g., 'BTC')
            timeframe: Timeframe (e.g., '1d')
            metrics: Model metrics dictionary
            output_dir: Output directory
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save model in .keras format
        model_filename = f"{symbol}_{timeframe}.keras"
        model_path = os.path.join(output_dir, model_filename)
        model.save(model_path, save_format='keras')
        print(f"  [Save] {model_filename} (format: .keras)")
        
        # Save metrics as JSON
        metrics_filename = f"{symbol}_{timeframe}_metrics.json"
        metrics_path = os.path.join(output_dir, metrics_filename)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  [Save] {metrics_filename}")
        
        return model_path, metrics_path

# ============================================================================
# HF UPLOAD (Batch Mode)
# ============================================================================

class HFUploader:
    def __init__(self, token: str, repo_id: str):
        self.token = token
        self.repo_id = repo_id
    
    def upload_after_training(self, symbol: str, timeframe: str):
        """
        Models are uploaded via batch upload script (upload_models_v6_batch.py)
        This method is for reference only.
        """
        print(f"  [Note] Use 'python upload_models_v6_batch.py' to batch upload all models")
        print(f"  [Note] HF Repository: https://huggingface.co/datasets/{self.repo_id}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Get HF token if in Colab
    if COLAB_MODE:
        try:
            from google.colab import userdata
            HF_TOKEN = userdata.get('HF_TOKEN')
            print(f"HF Token loaded from Colab Secrets")
        except:
            print("HF Token not found in Colab Secrets. Models will be saved locally.")
            HF_TOKEN = None
    
    fetcher = DataFetcher()
    trainer = Trainer(MODEL_PARAMS)
    evaluator = Evaluator()
    saver = ModelSaver()
    uploader = HFUploader(HF_TOKEN, HF_REPO_ID) if HF_TOKEN else None
    
    results_summary = {}
    
    # Iterate over all coins and timeframes
    total_tasks = len(COINS) * len(TIMEFRAMES)
    current_task = 0
    
    for coin_code, coin_yf in COINS.items():
        for timeframe, tf_config in TIMEFRAMES.items():
            current_task += 1
            print(f"\n[{current_task}/{total_tasks}] Training {coin_code} ({timeframe})")
            print("-" * 60)
            
            try:
                # 1. Fetch data
                df = fetcher.fetch(coin_yf, tf_config['period'], tf_config['interval'])
                if df is None or len(df) < MODEL_PARAMS['lookback'] + MODEL_PARAMS['forecast']:
                    print(f"  Insufficient data for {coin_code} ({timeframe}). Skipping.")
                    results_summary[f"{coin_code}_{timeframe}"] = "FAILED: Insufficient Data"
                    continue
                
                # 2. Prepare sequences
                print(f"  [Prep] Building sequences ({len(df)} bars)...", end=' ')
                seq_builder = SequenceBuilder(MODEL_PARAMS['lookback'], MODEL_PARAMS['forecast'])
                X, y, scalers = seq_builder.prepare_data(df)
                
                if X is None:
                    print("FAILED")
                    results_summary[f"{coin_code}_{timeframe}"] = "FAILED: Sequence Build"
                    continue
                
                print(f"OK ({X.shape[0]} sequences, {X.shape[2]} features)")
                
                # 3. Split data
                n_train = int(len(X) * (1 - MODEL_PARAMS['validation_split']))
                X_train, X_val = X[:n_train], X[n_train:]
                y_train, y_val = y[:n_train], y[n_train:]
                
                # 4. Train model
                print(f"  [Train] Training LSTM...")
                model = trainer.train(X_train, y_train, X_val, y_val, X.shape[2], verbose=0)
                print(f"  [Train] Complete")
                
                # 5. Evaluate
                feature_cols = ['Open', 'High', 'Low', 'Close']
                metrics = evaluator.evaluate(model, X_val, y_val, scalers, feature_cols)
                
                print(f"  [Metrics]")
                for col, m in metrics.items():
                    print(f"    {col}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, MAPE={m['MAPE']:.2f}%")
                
                # 6. Save model (.keras format)
                print(f"  [Save]")
                saver.save_local(model, coin_code, timeframe, metrics)
                
                results_summary[f"{coin_code}_{timeframe}"] = "SUCCESS"
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results_summary[f"{coin_code}_{timeframe}"] = f"FAILED: {e}"
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    success_count = sum(1 for v in results_summary.values() if v == "SUCCESS")
    failed_count = len(results_summary) - success_count
    
    print(f"\nSuccess: {success_count}/{total_tasks}")
    print(f"Failed: {failed_count}/{total_tasks}")
    
    print(f"\nDetailed Results:")
    for key, status in results_summary.items():
        symbol = '✓' if status == "SUCCESS" else '✗'
        print(f"  {symbol} {key}: {status}")
    
    print("\n" + "="*80)
    print(f"Models saved to: models_v6/ (.keras format)")
    print(f"\nNext Step: Batch Upload All Models")
    print(f"  Command: python upload_models_v6_batch.py")
    print(f"  This will upload all models to HF in one batch operation")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
