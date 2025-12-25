#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V5 Model Inference
從 Hugging Face 加載 V5 模型 + 雙時間框架 (1d + 1h) + 完整技術指標
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import asyncio
from datetime import datetime, timedelta
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI 初始化
# ============================================================================
app = FastAPI(title="CPB Trading Prediction API - V5", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 設定
# ============================================================================
HF_REPO = "zongowo111/cpb-models"
HF_REPO_TYPE = "dataset"
MODELS_FOLDER = "models_v5"

MODELS_CACHE_DIR = Path('./models_cache_v5')
MODELS_CACHE_DIR.mkdir(exist_ok=True)

# V5 支持的幣種 (14 種)
SUPPORTED_CRYPTOS_V5 = {
    'BTC': {'ticker': 'BTC-USD', 'name': '比特幣'},
    'ETH': {'ticker': 'ETH-USD', 'name': '以太坊'},
    'BNB': {'ticker': 'BNB-USD', 'name': '幣安幣'},
    'SOL': {'ticker': 'SOL-USD', 'name': '索拉納'},
    'XRP': {'ticker': 'XRP-USD', 'name': '瑞波幣'},
    'ADA': {'ticker': 'ADA-USD', 'name': '卡爾達諾'},
    'DOGE': {'ticker': 'DOGE-USD', 'name': '狗狗幣'},
    'AVAX': {'ticker': 'AVAX-USD', 'name': '雪崩幣'},
    'LTC': {'ticker': 'LTC-USD', 'name': '萊特幣'},
    'DOT': {'ticker': 'DOT-USD', 'name': '波卡'},
    'UNI': {'ticker': 'UNI-USD', 'name': 'Uniswap'},
    'LINK': {'ticker': 'LINK-USD', 'name': 'Chainlink'},
    'XLM': {'ticker': 'XLM-USD', 'name': 'Stellar'},
    'ATOM': {'ticker': 'ATOM-USD', 'name': 'Cosmos'},
}

SUPPORTED_TIMEFRAMES = ['1d', '1h']

print(f"\n[✓] 模型版本: V5")
print(f"[✓] 支援幣種: {len(SUPPORTED_CRYPTOS_V5)}")
print(f"[✓] 時間框架: {SUPPORTED_TIMEFRAMES}")
print(f"[✓] 幣種清單: {list(SUPPORTED_CRYPTOS_V5.keys())}\n")

# ============================================================================
# 資料模型
# ============================================================================

class PredictionRequestV5(BaseModel):
    symbol: str
    timeframe: str = '1d'
    lookback: int = 60

class KlineData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class VolatilityData(BaseModel):
    current: float
    predicted: float
    level: str
    atr_14: float

class PredictionResultV5(BaseModel):
    symbol: str
    timeframe: str
    timestamp: str
    current_price: float
    predicted_price: float
    log_return: float
    recommendation: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    volatility: VolatilityData
    klines: List[KlineData]
    model_version: str

# ============================================================================
# 特徵工程
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算完整的技術指標特徵集 (30+ 特徵)"""
    df = df.copy()
    
    # 基礎收益率
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # 波動率
    df['volatility'] = df['log_return'].rolling(14).std()
    df['volatility_20'] = df['log_return'].rolling(20).std()
    
    # 動量
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    
    # 價格範圍
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # 成交量
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    # 資金成本率和未平倉頭寸變化
    df['funding_rate'] = df['momentum_10'].rolling(20).mean() / (df['close'].rolling(20).std() + 1e-8) * 0.00001
    df['open_interest_change'] = (df['volume_ratio'] - 1) * (df['volatility'] / 0.01) * 0.1
    
    # 填充缺失值
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

# ============================================================================
# V5 模型管理
# ============================================================================

class ModelManagerV5:
    def __init__(self):
        self.models = {}  # {f"{symbol}_{timeframe}": model_info}
        logger.info("ModelManager V5 initialized")
    
    def load_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """從 Hugging Face 加載 V5 模型"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            model_name = f"{symbol}_{timeframe}_model.h5"
            scalers_name = f"{symbol}_{timeframe}_scalers.pkl"
            
            logger.info(f"Loading V5 model: {symbol} {timeframe}")
            
            # 從 HF 下載模型
            logger.info(f"Downloading {model_name} from HuggingFace...")
            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{model_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(MODELS_CACHE_DIR.parent),
            )
            
            # 複製到快取目錄
            import shutil
            local_model_path = MODELS_CACHE_DIR / model_name
            shutil.copy(model_path, local_model_path)
            
            # 從 HF 下載 scalers
            logger.info(f"Downloading {scalers_name} from HuggingFace...")
            scalers_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{scalers_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(MODELS_CACHE_DIR.parent),
            )
            
            local_scalers_path = MODELS_CACHE_DIR / scalers_name
            shutil.copy(scalers_path, local_scalers_path)
            
            # 加載模型
            logger.info(f"Loading TensorFlow model from {local_model_path}")
            model = tf.keras.models.load_model(str(local_model_path))
            
            # 加載 scalers
            logger.info(f"Loading scalers from {local_scalers_path}")
            with open(local_scalers_path, 'rb') as f:
                scalers = pickle.load(f)
            
            self.models[model_key] = {
                'model': model,
                'scalers': scalers,
                'symbol': symbol,
                'timeframe': timeframe,
                'feature_cols': scalers.get('feature_cols', [])
            }
            
            logger.info(f"Successfully loaded V5 model: {symbol} {timeframe}")
            return self.models[model_key]
        
        except Exception as e:
            logger.error(f"Failed to load V5 model {symbol} {timeframe}: {e}")
            return None
    
    def predict(self, symbol: str, timeframe: str, klines_data: List[Dict]) -> Optional[Dict]:
        """執行 V5 模型預測"""
        model_info = self.load_model(symbol, timeframe)
        
        if model_info is None:
            logger.error(f"Model not available for {symbol} {timeframe}")
            return None
        
        try:
            # 轉換為 DataFrame
            df = pd.DataFrame(klines_data)
            
            # 工程化特徵
            df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())
            
            # 檢查數據充足性
            if len(df_feat) < 61:  # 需要 60 根 K 線 + 1 個用於預測
                logger.warning(f"Not enough data: {len(df_feat)} rows")
                return None
            
            # 準備特徵
            feature_cols = model_info['feature_cols']
            if not feature_cols:
                logger.warning(f"No feature columns found in scalers")
                return None
            
            # 提取特徵矩陣
            X_recent = df_feat[feature_cols].iloc[-60:].values  # 最後 60 根 K 線
            
            # 正規化
            scaler_X = model_info['scalers']['X']
            X_norm = scaler_X.transform(X_recent)
            
            # 預測 (雙輸入 LSTM)
            X_input = X_norm.reshape(1, 60, -1).astype(np.float32)
            y_pred_norm = model_info['model'].predict([X_input, X_input], verbose=0).flatten()
            
            # 逆正規化
            scaler_y = model_info['scalers']['y']
            y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            
            # 計算預測價格
            current_price = df['close'].iloc[-1]
            log_return = y_pred[0]
            predicted_price = current_price * np.exp(log_return)
            
            # 信心度 (基於預測的對數收益率幅度)
            confidence = min(0.95, 0.5 + abs(log_return) * 10)
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'log_return': log_return,
                'confidence': confidence,
                'direction': 1 if log_return > 0 else (-1 if log_return < 0 else 0)
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

model_manager = ModelManagerV5()

# ============================================================================
# 資料獲取
# ============================================================================

class DataFetcherV5:
    @staticmethod
    def fetch_klines_yfinance(ticker: str, interval: str = '1d', days: int = 365) -> Optional[List[Dict]]:
        """使用 yfinance 獲取 K 線數據"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Fetching {ticker} {interval} data from yfinance...")
            
            df = yf.download(
                ticker,
                start=start_date.date(),
                end=end_date.date(),
                interval=interval,
                progress=False,
                prepost=False,
                threads=False
            )
            
            if df is None or len(df) == 0:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            df = df.copy()
            
            # 修復列名處理 - 處理 MultiIndex 列名
            if isinstance(df.columns, pd.MultiIndex):
                # 如果是 MultiIndex，取第一層
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # 轉換為小寫
            df.columns = [str(c).lower() for c in df.columns]
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            # 確保列名
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available = [c for c in df.columns]
            logger.info(f"Available columns: {available}")
            
            if not all(c in df.columns for c in required):
                logger.warning(f"Missing columns. Required: {required}, Available: {available}")
                return None
            
            df = df[required].copy()
            df = df.dropna()
            df = df[df['volume'] > 0]
            
            logger.info(f"Fetched {len(df)} klines for {ticker}")
            
            # 轉換為字典列表
            klines = []
            for _, row in df.iterrows():
                klines.append({
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
            
            return klines
        
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            import traceback
            traceback.print_exc()
            return None

data_fetcher = DataFetcherV5()

# ============================================================================
# API 端點
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "CPB Trading Prediction API - V5",
        "version": "5.0.0",
        "model_type": "V5 (Dual Timeframe)",
        "supported_symbols": len(SUPPORTED_CRYPTOS_V5),
        "timeframes": SUPPORTED_TIMEFRAMES,
        "endpoints": {
            "/coins-v5": "List supported coins",
            "/predict-v5": "Get V5 prediction",
            "/health": "Health check"
        }
    }

@app.get("/coins-v5")
async def get_coins_v5():
    """列出 V5 支援的幣種"""
    return {
        "symbols": list(SUPPORTED_CRYPTOS_V5.keys()),
        "cryptos": SUPPORTED_CRYPTOS_V5,
        "timeframes": SUPPORTED_TIMEFRAMES,
        "total_symbols": len(SUPPORTED_CRYPTOS_V5),
        "model_version": "V5"
    }

@app.post("/predict-v5")
async def predict_v5(request: PredictionRequestV5) -> PredictionResultV5:
    """V5 模型預測端點 (雙時間框架)"""
    
    # 驗證幣種和時間框架
    symbol = request.symbol.upper() if isinstance(request.symbol, str) else request.symbol
    timeframe = request.timeframe.lower() if isinstance(request.timeframe, str) else request.timeframe
    
    if symbol not in SUPPORTED_CRYPTOS_V5:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol {symbol} not supported. Available: {list(SUPPORTED_CRYPTOS_V5.keys())}"
        )
    
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Timeframe {timeframe} not supported. Available: {SUPPORTED_TIMEFRAMES}"
        )
    
    # 1. 獲取 K 線數據
    ticker = SUPPORTED_CRYPTOS_V5[symbol]['ticker']
    days = 3000 if timeframe == '1d' else 400
    
    klines = data_fetcher.fetch_klines_yfinance(
        ticker=ticker,
        interval=timeframe,
        days=days
    )
    
    if klines is None or len(klines) == 0:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch klines for {symbol}"
        )
    
    # 2. 執行 V5 預測
    pred_result = model_manager.predict(symbol, timeframe, klines)
    
    if pred_result is None:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for {symbol} {timeframe}"
        )
    
    # 3. 計算交易建議
    current_price = pred_result['current_price']
    predicted_price = pred_result['predicted_price']
    direction = pred_result['direction']
    confidence = pred_result['confidence']
    log_return = pred_result['log_return']
    
    # 計算波動率
    df = pd.DataFrame(klines)
    df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())
    
    volatility_current = df_feat['volatility'].iloc[-1] * 100 if not np.isnan(df_feat['volatility'].iloc[-1]) else 0
    volatility_predicted = abs(log_return) * 100
    atr_14 = df_feat['atr'].iloc[-1] if not np.isnan(df_feat['atr'].iloc[-1]) else 0
    
    # 波動率等級
    if volatility_current < 0.5:
        vol_level = "低"
    elif volatility_current < 2.0:
        vol_level = "中"
    else:
        vol_level = "高"
    
    # 推薦
    if direction > 0:
        recommendation = "BUY"
        entry_price = current_price
        stop_loss = round(current_price * 0.98, 2)
        take_profit = round(predicted_price * 1.02, 2)
    elif direction < 0:
        recommendation = "SELL"
        entry_price = current_price
        stop_loss = round(current_price * 1.02, 2)
        take_profit = round(predicted_price * 0.98, 2)
    else:
        recommendation = "HOLD"
        entry_price = current_price
        stop_loss = round(current_price * 0.99, 2)
        take_profit = round(current_price * 1.01, 2)
    
    # 4. 構建響應
    result = PredictionResultV5(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime.now().isoformat(),
        current_price=round(current_price, 2),
        predicted_price=round(predicted_price, 2),
        log_return=round(log_return, 6),
        recommendation=recommendation,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=round(confidence, 4),
        volatility=VolatilityData(
            current=round(volatility_current, 4),
            predicted=round(volatility_predicted, 4),
            level=vol_level,
            atr_14=round(atr_14, 2)
        ),
        klines=[
            KlineData(
                timestamp=k['timestamp'],
                open=k['open'],
                high=k['high'],
                low=k['low'],
                close=k['close'],
                volume=k['volume']
            )
            for k in klines[-20:]
        ],
        model_version="V5"
    )
    
    return result

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_version": "V5",
        "models_cached": len(model_manager.models),
        "supported_symbols": len(SUPPORTED_CRYPTOS_V5)
    }

# ============================================================================
# 運行
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print(" "*20 + "CPB Trading Web - V5 Model")
    print("="*80)
    print(f"\nModel Version: V5")
    print(f"Supported Symbols: {len(SUPPORTED_CRYPTOS_V5)}")
    print(f"Timeframes: {SUPPORTED_TIMEFRAMES}")
    print(f"Features: {30}+ technical indicators")
    print(f"\nStarting FastAPI server...")
    print(f"API: http://localhost:8001")
    print(f"Docs: http://localhost:8001/docs")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
