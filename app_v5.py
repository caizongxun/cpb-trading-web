#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V5 Model Inference (HYBRID VERSION)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import warnings

# 抑制 TensorFlow 警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPB Trading Prediction API - V5", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_REPO = "zongowo111/cpb-models"
HF_REPO_TYPE = "dataset"
MODELS_FOLDER = "models_v5"
MODELS_CACHE_DIR = Path('./models_cache_v5')
MODELS_CACHE_DIR.mkdir(exist_ok=True)

SUPPORTED_CRYPTOS_V5 = {
    'BTC': {'ticker': 'BTC-USD', 'binance': 'BTCUSDT', 'name': '比特幣'},
    'ETH': {'ticker': 'ETH-USD', 'binance': 'ETHUSDT', 'name': '以太坊'},
    'BNB': {'ticker': 'BNB-USD', 'binance': 'BNBUSDT', 'name': '幣安幣'},
    'SOL': {'ticker': 'SOL-USD', 'binance': 'SOLUSDT', 'name': '索拉納'},
    'XRP': {'ticker': 'XRP-USD', 'binance': 'XRPUSDT', 'name': '瑞波幣'},
    'ADA': {'ticker': 'ADA-USD', 'binance': 'ADAUSDT', 'name': '卡爾達諾'},
    'DOGE': {'ticker': 'DOGE-USD', 'binance': 'DOGEUSDT', 'name': '狗狗幣'},
    'AVAX': {'ticker': 'AVAX-USD', 'binance': 'AVAXUSDT', 'name': '雪崩幣'},
    'LTC': {'ticker': 'LTC-USD', 'binance': 'LTCUSDT', 'name': '萊特幣'},
    'DOT': {'ticker': 'DOT-USD', 'binance': 'DOTUSDT', 'name': '波卡'},
    'UNI': {'ticker': 'UNI-USD', 'binance': 'UNIUSDT', 'name': 'Uniswap'},
    'LINK': {'ticker': 'LINK-USD', 'binance': 'LINKUSDT', 'name': 'Chainlink'},
    'XLM': {'ticker': 'XLM-USD', 'binance': 'XLMUSDT', 'name': 'Stellar'},
    'ATOM': {'ticker': 'ATOM-USD', 'binance': 'ATOMUSDT', 'name': 'Cosmos'},
}

SUPPORTED_TIMEFRAMES = ['1d', '1h']

print(f"\n[✓] 模型版本: V5 (HYBRID)")
print(f"[✓] 支援幣種: {len(SUPPORTED_CRYPTOS_V5)}")
print(f"[✓] 時間框架: {SUPPORTED_TIMEFRAMES}")
print(f"[✓] 價格源: yfinance (統一當前價格)")
print(f"[✓] Binance 支援: 已內置 (可選使用)")
print(f"[✓] TensorFlow 已配置\n")

class PredictionRequestV5(BaseModel):
    symbol: str
    timeframe: str = '1d'
    lookback: int = 60
    use_binance: bool = False

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
    price_source: str

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volatility'] = df['log_return'].rolling(14).std()
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    df['funding_rate'] = df['momentum_10'].rolling(20).mean() / (df['close'].rolling(20).std() + 1e-8) * 0.00001
    df['open_interest_change'] = (df['volume_ratio'] - 1) * (df['volatility'] / 0.01) * 0.1
    
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], 0)
    return df

class ModelManagerV5:
    def __init__(self):
        self.models = {}
        self.failed_models = set()
        logger.info("ModelManager V5 (HYBRID) initialized")
    
    def try_load_real_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        model_key = f"{symbol}_{timeframe}"
        try:
            model_name = f"{symbol}_{timeframe}_model.h5"
            scalers_name = f"{symbol}_{timeframe}_scalers.pkl"
            
            logger.info(f"[REAL] Loading V5 model: {symbol} {timeframe}")
            
            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{model_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(MODELS_CACHE_DIR.parent),
            )
            
            import shutil
            local_model_path = MODELS_CACHE_DIR / model_name
            shutil.copy(model_path, local_model_path)
            
            scalers_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{scalers_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(MODELS_CACHE_DIR.parent),
            )
            
            local_scalers_path = MODELS_CACHE_DIR / scalers_name
            shutil.copy(scalers_path, local_scalers_path)
            
            logger.info(f"[REAL] Loading TensorFlow model...")
            
            # 修復: 移除已棄用的 set_learning_phase API
            try:
                model = tf.keras.models.load_model(
                    str(local_model_path),
                    compile=False  # 不編譯以避免相容性問題
                )
                model.compile(optimizer='adam', loss='mse')
            except Exception as e:
                logger.warning(f"[REAL] Model compile error, trying alternative load: {str(e)[:80]}")
                model = tf.keras.models.load_model(str(local_model_path))
            
            with open(local_scalers_path, 'rb') as f:
                scalers = pickle.load(f)
            
            logger.info(f"[REAL] Model loaded successfully")
            return {
                'model': model,
                'scalers': scalers,
                'symbol': symbol,
                'timeframe': timeframe,
                'feature_cols': scalers.get('feature_cols', []),
                'is_real': True
            }
        except Exception as e:
            logger.warning(f"[REAL] Failed to load real model: {str(e)[:100]}")
            return None
    
    def predict_real(self, model_info: Dict, klines_data: List[Dict]) -> Optional[Dict]:
        try:
            df = pd.DataFrame(klines_data)
            df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())
            
            if len(df_feat) < 61:
                return None
            
            feature_cols = model_info['feature_cols']
            if not feature_cols:
                return None
            
            X_recent = df_feat[feature_cols].iloc[-60:].values
            scaler_X = model_info['scalers']['X']
            X_norm = scaler_X.transform(X_recent)
            X_input = X_norm.reshape(1, 60, -1).astype(np.float32)
            
            # 使用 predict 時不需要特殊的 training 參數
            y_pred_norm = model_info['model'].predict(X_input, verbose=0).flatten()
            
            scaler_y = model_info['scalers']['y']
            y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            
            current_price = df['close'].iloc[-1]
            log_return = y_pred[0]
            predicted_price = current_price * np.exp(log_return)
            confidence = min(0.95, 0.5 + abs(log_return) * 10)
            
            logger.info(f"[REAL] Prediction successful")
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'log_return': log_return,
                'confidence': confidence,
                'direction': 1 if log_return > 0 else (-1 if log_return < 0 else 0),
                'method': 'REAL_MODEL'
            }
        except Exception as e:
            logger.warning(f"[REAL] Prediction failed: {str(e)[:100]}")
            return None
    
    def predict_demo(self, symbol: str, timeframe: str, klines_data: List[Dict]) -> Optional[Dict]:
        try:
            df = pd.DataFrame(klines_data)
            df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())
            
            if len(df_feat) < 61:
                return None
            
            current_price = df['close'].iloc[-1]
            recent_closes = df['close'].iloc[-60:].values
            returns = np.log(recent_closes[1:] / recent_closes[:-1])
            volatility = np.std(returns)
            trend = np.mean(returns)
            
            rsi = df_feat['rsi'].iloc[-1]
            if np.isnan(rsi):
                rsi = 50
            
            macd = df_feat['macd'].iloc[-1]
            macd_signal = df_feat['macd_signal'].iloc[-1]
            if np.isnan(macd):
                macd = 0
            if np.isnan(macd_signal):
                macd_signal = 0
            
            base_log_return = trend + volatility * 0.5
            
            if rsi > 70:
                base_log_return *= 0.7
            elif rsi < 30:
                base_log_return *= 1.3
            
            if macd > macd_signal:
                base_log_return += abs(macd - macd_signal) * 0.01
            else:
                base_log_return -= abs(macd - macd_signal) * 0.01
            
            noise = np.random.normal(0, volatility * 0.3)
            log_return = base_log_return + noise
            
            if timeframe == '1d':
                log_return = np.clip(log_return, -0.02, 0.02)
            else:
                log_return = np.clip(log_return, -0.01, 0.01)
            
            predicted_price = current_price * np.exp(log_return)
            
            signals = 0
            if log_return > 0 and rsi < 70:
                signals += 1
            if log_return > 0 and macd > macd_signal:
                signals += 1
            if log_return < 0 and rsi > 30:
                signals += 1
            if log_return < 0 and macd < macd_signal:
                signals += 1
            
            confidence = 0.5 + (signals / 8) * 0.45
            confidence = np.clip(confidence, 0.4, 0.95)
            
            logger.info(f"[DEMO] Fallback prediction successful")
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'log_return': log_return,
                'confidence': confidence,
                'direction': 1 if log_return > 0.001 else (-1 if log_return < -0.001 else 0),
                'method': 'DEMO_FALLBACK'
            }
        except Exception as e:
            logger.error(f"[DEMO] Fallback failed: {str(e)[:100]}")
            return None
    
    def predict(self, symbol: str, timeframe: str, klines_data: List[Dict]) -> Optional[Dict]:
        model_key = f"{symbol}_{timeframe}"
        
        if model_key in self.failed_models:
            logger.info(f"[HYBRID] {model_key} previously failed, using DEMO")
            return self.predict_demo(symbol, timeframe, klines_data)
        
        if model_key not in self.models:
            model_info = self.try_load_real_model(symbol, timeframe)
            if model_info:
                self.models[model_key] = model_info
            else:
                self.failed_models.add(model_key)
                logger.info(f"[HYBRID] Real model failed for {model_key}, using DEMO")
        
        if model_key in self.models:
            result = self.predict_real(self.models[model_key], klines_data)
            if result:
                return result
            else:
                logger.info(f"[HYBRID] Real model prediction failed, falling back to DEMO")
                self.failed_models.add(model_key)
        
        return self.predict_demo(symbol, timeframe, klines_data)

model_manager = ModelManagerV5()

class DataFetcherV5:
    @staticmethod
    def fetch_klines_yfinance(ticker: str, interval: str = '1d', days: int = 365) -> Optional[List[Dict]]:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"[yfinance] Fetching {ticker} {interval} data...")
            
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
                logger.warning(f"[yfinance] No data returned for {ticker}")
                return None
            
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            df.columns = [str(c).lower() for c in df.columns]
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required):
                logger.warning(f"[yfinance] Missing columns. Available: {df.columns.tolist()}")
                return None
            
            df = df[required].copy()
            df = df.dropna()
            df = df[df['volume'] > 0]
            
            logger.info(f"[yfinance] Fetched {len(df)} klines for {ticker}")
            
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
            logger.error(f"[yfinance] Error: {e}")
            return None
    
    @staticmethod
    def fetch_klines_binance(symbol: str, interval: str = '1d', limit: int = 1000) -> Optional[List[Dict]]:
        try:
            import ccxt
            
            logger.info(f"[Binance] Fetching {symbol} {interval} data...")
            exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
            
            timeframe_map = {'1d': '1d', '1h': '1h'}
            tf = timeframe_map.get(interval, '1d')
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            
            klines = []
            for candle in ohlcv:
                timestamp = datetime.fromtimestamp(candle[0] / 1000).isoformat()
                klines.append({
                    'timestamp': timestamp,
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            logger.info(f"[Binance] Fetched {len(klines)} klines for {symbol}")
            return klines
        except ImportError:
            logger.error("[Binance] ccxt not installed. Use: pip install ccxt")
            return None
        except Exception as e:
            logger.error(f"[Binance] Error: {e}")
            return None

data_fetcher = DataFetcherV5()

@app.get("/")
async def root():
    return {
        "message": "CPB Trading Prediction API - V5",
        "version": "5.0.0",
        "model_type": "V5 (Hybrid: Real Model + Demo Fallback)",
        "supported_symbols": len(SUPPORTED_CRYPTOS_V5),
        "timeframes": SUPPORTED_TIMEFRAMES,
        "price_sources": ["yfinance", "binance (optional)"],
    }

@app.get("/coins-v5")
async def get_coins_v5():
    return {
        "symbols": list(SUPPORTED_CRYPTOS_V5.keys()),
        "cryptos": SUPPORTED_CRYPTOS_V5,
        "timeframes": SUPPORTED_TIMEFRAMES,
        "total_symbols": len(SUPPORTED_CRYPTOS_V5),
        "model_version": "V5"
    }

@app.post("/predict-v5")
async def predict_v5(request: PredictionRequestV5) -> PredictionResultV5:
    symbol = request.symbol.upper() if isinstance(request.symbol, str) else request.symbol
    timeframe = request.timeframe.lower() if isinstance(request.timeframe, str) else request.timeframe
    use_binance = request.use_binance if hasattr(request, 'use_binance') else False
    
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
    
    if use_binance:
        binance_symbol = SUPPORTED_CRYPTOS_V5[symbol]['binance']
        days = 3000 if timeframe == '1d' else 400
        limit = (days * 24) // (1 if timeframe == '1d' else 24)
        klines = data_fetcher.fetch_klines_binance(symbol=binance_symbol, interval=timeframe, limit=min(limit, 1000))
        price_source = "Binance"
    else:
        ticker = SUPPORTED_CRYPTOS_V5[symbol]['ticker']
        days = 3000 if timeframe == '1d' else 400
        klines = data_fetcher.fetch_klines_yfinance(ticker=ticker, interval=timeframe, days=days)
        price_source = "yfinance"
    
    if klines is None or len(klines) == 0:
        raise HTTPException(status_code=500, detail=f"Failed to fetch klines for {symbol} from {price_source}")
    
    pred_result = model_manager.predict(symbol, timeframe, klines)
    
    if pred_result is None:
        raise HTTPException(status_code=500, detail=f"Prediction failed for {symbol} {timeframe}")
    
    current_price = pred_result['current_price']
    predicted_price = pred_result['predicted_price']
    direction = pred_result['direction']
    confidence = pred_result['confidence']
    log_return = pred_result['log_return']
    prediction_method = pred_result.get('method', 'UNKNOWN')
    
    df = pd.DataFrame(klines)
    df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())
    
    volatility_current = df_feat['volatility'].iloc[-1] * 100 if not np.isnan(df_feat['volatility'].iloc[-1]) else 0
    volatility_predicted = abs(log_return) * 100
    atr_14 = df_feat['atr'].iloc[-1] if not np.isnan(df_feat['atr'].iloc[-1]) else 0
    
    if volatility_current < 0.5:
        vol_level = "低"
    elif volatility_current < 2.0:
        vol_level = "中"
    else:
        vol_level = "高"
    
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
    
    model_version = f"V5-{prediction_method}"
    
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
        klines=[KlineData(timestamp=k['timestamp'], open=k['open'], high=k['high'], low=k['low'], close=k['close'], volume=k['volume']) for k in klines[-20:]],
        model_version=model_version,
        price_source=price_source
    )
    
    return result

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_version": "V5-HYBRID",
        "supported_symbols": len(SUPPORTED_CRYPTOS_V5),
        "price_sources": ["yfinance (default)", "binance (optional)"]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print(" "*15 + "CPB Trading Web - V5 Model (HYBRID VERSION)")
    print("="*80)
    print(f"\nModel Version: V5 (HYBRID)")
    print(f"Strategy: Try Real Model → Fallback to Demo Predictions")
    print(f"Price Source: yfinance (unified, consistent across timeframes)")
    print(f"Supported Symbols: {len(SUPPORTED_CRYPTOS_V5)}")
    print(f"Timeframes: {SUPPORTED_TIMEFRAMES}")
    print(f"\nStarting FastAPI server...")
    print(f"API: http://localhost:8001")
    print(f"Docs: http://localhost:8001/docs")
    print(f"\n⚠  HYBRID MODE: Always provides predictions!")
    print(f"   - If real model loads: Uses trained model")
    print(f"   - If real model fails: Falls back to intelligent demo predictions")
    print(f"\n⚠  PRICE CONSISTENCY:")
    print(f"   - 1D and 1H use SAME current price (from yfinance)")
    print(f"   - Binance optional (use_binance=true parameter)")
    print(f"\n⚠  TENSORFLOW FIX:")
    print(f"   - Removed deprecated set_learning_phase API")
    print(f"   - Using compile=False for model loading")
    print(f"\n" + "="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
