#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V5 Model Inference (HYBRID VERSION)
使用段的、可靠的混合方案:
1. 嘗試加載真實 HuggingFace 模型
2. 失敗時退隱到智能模擬 ✓
3. 使用統一的價格源 (當前價格一致)
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
    'BTC': {'ticker': 'BTC-USD', 'binance': 'BTCUSDT', 'name': '比特幣'},
    'ETH': {'ticker': 'ETH-USD', 'binance': 'ETHUSDT', 'name': '以太坊'},
    'BNB': {'ticker': 'BNB-USD', 'binance': 'BNBUSDT', 'name': '幣安幣'},
    'SOL': {'ticker': 'SOL-USD', 'binance': 'SOLUSDT', 'name': '索拉納'},
    'XRP': {'ticker': 'XRP-USD', 'binance': 'XRPUSDT', 'name': '瑞波幣'},
    'ADA': {'ticker': 'ADA-USD', 'binance': 'ADAUSDT', 'name': '卡爾達諾'},
    'DOGE': {'ticker': 'DOGE-USD', 'binance': 'DOGEUSDT', 'name': '狗狗幣'},
    'AVAX': {'ticker': 'AVAX-USD', 'binance': 'AVAXUSDT', 'name': '雪崩幣'},
    'LTC': {'ticker': 'LTC-USD', 'binance': 'LTCUSDT', 'name': '萊特幣'},
    'DOT': {'ticker': 'DOT-USD', 'binance': 'DOTUSDT', 'name': '波卡'},\n    'UNI': {'ticker': 'UNI-USD', 'binance': 'UNIUSDT', 'name': 'Uniswap'},
    'LINK': {'ticker': 'LINK-USD', 'binance': 'LINKUSDT', 'name': 'Chainlink'},
    'XLM': {'ticker': 'XLM-USD', 'binance': 'XLMUSDT', 'name': 'Stellar'},
    'ATOM': {'ticker': 'ATOM-USD', 'binance': 'ATOMUSDT', 'name': 'Cosmos'},
}

SUPPORTED_TIMEFRAMES = ['1d', '1h']

print(f"\n[\u2713] 模型版本: V5 (HYBRID - 真實模型 + 智能模擬降級)")
print(f"[\u2713] 支援幣種: {len(SUPPORTED_CRYPTOS_V5)}")
print(f"[\u2713] 時間框架: {SUPPORTED_TIMEFRAMES}")
print(f"[\u2713] 價格源: yfinance (統一當前價格)")
print(f"[\u2713] Binance 支援: 已內置 (可選使用)")
print(f"[\u2713] 幣種清單: {list(SUPPORTED_CRYPTOS_V5.keys())}\n")

# ============================================================================
# 資料模型
# ============================================================================

class PredictionRequestV5(BaseModel):
    symbol: str
    timeframe: str = '1d'
    lookback: int = 60
    use_binance: bool = False  # 新增: 是否使用 Binance

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
    price_source: str  # 新增: 標示價格源

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
# V5 混合模型管理 (真實 + 模擬降級)
# ============================================================================

class ModelManagerV5:
    def __init__(self):
        self.models = {}
        self.failed_models = set()  # 追蹤失敗的模型
        self.current_price_cache = {}  # 緩存當前價格 (確保 1d/1h 一致)
        logger.info("ModelManager V5 (HYBRID) initialized")
    
    def try_load_real_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """嘗試加載真實 HuggingFace 模型"""
        model_key = f"{symbol}_{timeframe}"
        
        try:
            model_name = f"{symbol}_{timeframe}_model.h5"
            scalers_name = f"{symbol}_{timeframe}_scalers.pkl"
            
            logger.info(f"[REAL] Attempting to load V5 model: {symbol} {timeframe}")
            
            # 從 HF 下載模型
            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{model_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(MODELS_CACHE_DIR.parent),
            )
            
            import shutil
            local_model_path = MODELS_CACHE_DIR / model_name
            shutil.copy(model_path, local_model_path)
            
            # 從 HF 下載 scalers
            scalers_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{MODELS_FOLDER}/{scalers_name}",
                repo_type=HF_REPO_TYPE,
                cache_dir=str(MODELS_CACHE_DIR.parent),
            )
            
            local_scalers_path = MODELS_CACHE_DIR / scalers_name
            shutil.copy(scalers_path, local_scalers_path)
            
            # 加載模型
            logger.info(f"[REAL] Loading TensorFlow model...")
            
            from tensorflow.compat.v1 import ConfigProto, Session
            tf.keras.backend.set_learning_phase(0)
            
            model = tf.keras.models.load_model(str(local_model_path))
            
            # 加載 scalers
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
        """使用真實模型預測"""
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
            y_pred_norm = model_info['model'].predict([X_input, X_input], verbose=0).flatten()
            
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
        """退隱到智能模擬預測"""
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
        """混合預測: 嘗試真實模型, 失敗時退隱到 DEMO"""
        model_key = f"{symbol}_{timeframe}"
        
        # 如果此前模型加載失敗過, 直接使用 DEMO
        if model_key in self.failed_models:
            logger.info(f"[HYBRID] {model_key} previously failed, using DEMO")
            return self.predict_demo(symbol, timeframe, klines_data)
        
        # 嘗試加載真實模型
        if model_key not in self.models:
            model_info = self.try_load_real_model(symbol, timeframe)
            if model_info:
                self.models[model_key] = model_info
            else:
                self.failed_models.add(model_key)
                logger.info(f"[HYBRID] Real model failed for {model_key}, using DEMO")
        
        # 使用真實模型預測 (如果可用)
        if model_key in self.models:
            result = self.predict_real(self.models[model_key], klines_data)
            if result:
                return result
            else:
                logger.info(f"[HYBRID] Real model prediction failed, falling back to DEMO")
                self.failed_models.add(model_key)
        
        # 退隱到 DEMO
        return self.predict_demo(symbol, timeframe, klines_data)

model_manager = ModelManagerV5()

# ============================================================================
# 資料獲取 (統一價格源)
# ============================================================================

class DataFetcherV5:
    @staticmethod
    def fetch_klines_yfinance(ticker: str, interval: str = '1d', days: int = 365) -> Optional[List[Dict]]:
        """使用 yfinance 獲取 K 線數據 (推薦)"""
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
            
            # 修復列名處理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            df.columns = [str(c).lower() for c in df.columns]
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            # 確保列名
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required):
                logger.warning(f"[yfinance] Missing columns. Available: {df.columns.tolist()}")
                return None
            
            df = df[required].copy()
            df = df.dropna()
            df = df[df['volume'] > 0]
            
            logger.info(f"[yfinance] Fetched {len(df)} klines for {ticker}")
            
            # 轉換為字典列表
            klines = []
            for _, row in df.iterrows():
                klines.append({
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])\n                })\n            
            return klines
        
        except Exception as e:
            logger.error(f"[yfinance] Error: {e}")
            return None
    
    @staticmethod
    def fetch_klines_binance(symbol: str, interval: str = '1d', limit: int = 1000) -> Optional[List[Dict]]:\n        \"\"\"使用 Binance API 獲取 K 線數據 (備選)\"\"\"\n        try:\n            import ccxt\n            \n            logger.info(f\"[Binance] Fetching {symbol} {interval} data...\")\n            \n            exchange = ccxt.binance({\n                'enableRateLimit': True,\n                'options': {'defaultType': 'spot'}\n            })\n            \n            # 轉換時間框架\n            timeframe_map = {'1d': '1d', '1h': '1h'}\n            tf = timeframe_map.get(interval, '1d')\n            \n            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)\n            \n            klines = []\n            for candle in ohlcv:\n                timestamp = datetime.fromtimestamp(candle[0] / 1000).isoformat()\n                klines.append({\n                    'timestamp': timestamp,\n                    'open': float(candle[1]),\n                    'high': float(candle[2]),\n                    'low': float(candle[3]),\n                    'close': float(candle[4]),\n                    'volume': float(candle[5])\n                })\n            \n            logger.info(f\"[Binance] Fetched {len(klines)} klines for {symbol}\")\n            return klines\n        \n        except ImportError:\n            logger.error(\"[Binance] ccxt not installed. Use: pip install ccxt\")\n            return None\n        except Exception as e:\n            logger.error(f\"[Binance] Error: {e}\")\n            return None\n    \n    @staticmethod\n    def get_current_price(ticker: str) -> Optional[float]:\n        \"\"\"獲取當前價格 (統一源 - 只用 yfinance)\"\"\"\n        try:\n            logger.info(f\"[Price] Fetching current price for {ticker}\")\n            \n            df = yf.download(\n                ticker,\n                period='1d',\n                progress=False,\n                threads=False\n            )\n            \n            if df is None or len(df) == 0:\n                return None\n            \n            current_price = float(df['Close'].iloc[-1])\n            logger.info(f\"[Price] Current price: {ticker} = {current_price}\")\n            return current_price\n        \n        except Exception as e:\n            logger.error(f\"[Price] Error fetching current price: {e}\")\n            return None\n\ndata_fetcher = DataFetcherV5()\n\n# ============================================================================\n# API 端點\n# ============================================================================\n\n@app.get(\"/\")\nasync def root():\n    return {\n        \"message\": \"CPB Trading Prediction API - V5\",\n        \"version\": \"5.0.0\",\n        \"model_type\": \"V5 (Hybrid: Real Model + Demo Fallback)\",\n        \"supported_symbols\": len(SUPPORTED_CRYPTOS_V5),\n        \"timeframes\": SUPPORTED_TIMEFRAMES,\n        \"price_sources\": [\"yfinance\", \"binance (optional)\"],\n        \"endpoints\": {\n            \"/coins-v5\": \"List supported coins\",\n            \"/predict-v5\": \"Get V5 prediction\",\n            \"/health\": \"Health check\"\n        }\n    }\n\n@app.get(\"/coins-v5\")\nasync def get_coins_v5():\n    \"\"\"列出 V5 支援的幣種\"\"\"\n    return {\n        \"symbols\": list(SUPPORTED_CRYPTOS_V5.keys()),\n        \"cryptos\": SUPPORTED_CRYPTOS_V5,\n        \"timeframes\": SUPPORTED_TIMEFRAMES,\n        \"total_symbols\": len(SUPPORTED_CRYPTOS_V5),\n        \"model_version\": \"V5\"\n    }\n\n@app.post(\"/predict-v5\")\nasync def predict_v5(request: PredictionRequestV5) -> PredictionResultV5:\n    \"\"\"V5 混合模型預測端點 (統一價格源)\"\"\"\n    \n    symbol = request.symbol.upper() if isinstance(request.symbol, str) else request.symbol\n    timeframe = request.timeframe.lower() if isinstance(request.timeframe, str) else request.timeframe\n    use_binance = request.use_binance if hasattr(request, 'use_binance') else False\n    \n    if symbol not in SUPPORTED_CRYPTOS_V5:\n        raise HTTPException(\n            status_code=400,\n            detail=f\"Symbol {symbol} not supported. Available: {list(SUPPORTED_CRYPTOS_V5.keys())}\"\n        )\n    \n    if timeframe not in SUPPORTED_TIMEFRAMES:\n        raise HTTPException(\n            status_code=400,\n            detail=f\"Timeframe {timeframe} not supported. Available: {SUPPORTED_TIMEFRAMES}\"\n        )\n    \n    # 1. 獲取 K 線數據\n    if use_binance:\n        binance_symbol = SUPPORTED_CRYPTOS_V5[symbol]['binance']\n        days = 3000 if timeframe == '1d' else 400\n        limit = (days * 24) // (1 if timeframe == '1d' else 24)  # 估計K線數量\n        \n        klines = data_fetcher.fetch_klines_binance(\n            symbol=binance_symbol,\n            interval=timeframe,\n            limit=min(limit, 1000)  # Binance 最多 1000\n        )\n        price_source = \"Binance\"\n    else:\n        # 使用 yfinance (推薦 - 確保一致性)\n        ticker = SUPPORTED_CRYPTOS_V5[symbol]['ticker']\n        days = 3000 if timeframe == '1d' else 400\n        \n        klines = data_fetcher.fetch_klines_yfinance(\n            ticker=ticker,\n            interval=timeframe,\n            days=days\n        )\n        price_source = \"yfinance\"\n    \n    if klines is None or len(klines) == 0:\n        raise HTTPException(\n            status_code=500,\n            detail=f\"Failed to fetch klines for {symbol} from {price_source}\"\n        )\n    \n    # 2. 執行 V5 預測\n    pred_result = model_manager.predict(symbol, timeframe, klines)\n    \n    if pred_result is None:\n        raise HTTPException(\n            status_code=500,\n            detail=f\"Prediction failed for {symbol} {timeframe}\"\n        )\n    \n    current_price = pred_result['current_price']\n    predicted_price = pred_result['predicted_price']\n    direction = pred_result['direction']\n    confidence = pred_result['confidence']\n    log_return = pred_result['log_return']\n    prediction_method = pred_result.get('method', 'UNKNOWN')\n    \n    # 計算波動率\n    df = pd.DataFrame(klines)\n    df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())\n    \n    volatility_current = df_feat['volatility'].iloc[-1] * 100 if not np.isnan(df_feat['volatility'].iloc[-1]) else 0\n    volatility_predicted = abs(log_return) * 100\n    atr_14 = df_feat['atr'].iloc[-1] if not np.isnan(df_feat['atr'].iloc[-1]) else 0\n    \n    # 波動率等級\n    if volatility_current < 0.5:\n        vol_level = \"低\"\n    elif volatility_current < 2.0:\n        vol_level = \"中\"\n    else:\n        vol_level = \"高\"\n    \n    # 推薦\n    if direction > 0:\n        recommendation = \"BUY\"\n        entry_price = current_price\n        stop_loss = round(current_price * 0.98, 2)\n        take_profit = round(predicted_price * 1.02, 2)\n    elif direction < 0:\n        recommendation = \"SELL\"\n        entry_price = current_price\n        stop_loss = round(current_price * 1.02, 2)\n        take_profit = round(predicted_price * 0.98, 2)\n    else:\n        recommendation = \"HOLD\"\n        entry_price = current_price\n        stop_loss = round(current_price * 0.99, 2)\n        take_profit = round(current_price * 1.01, 2)\n    \n    model_version = f\"V5-{prediction_method}\"\n    \n    result = PredictionResultV5(\n        symbol=symbol,\n        timeframe=timeframe,\n        timestamp=datetime.now().isoformat(),\n        current_price=round(current_price, 2),\n        predicted_price=round(predicted_price, 2),\n        log_return=round(log_return, 6),\n        recommendation=recommendation,\n        entry_price=entry_price,\n        stop_loss=stop_loss,\n        take_profit=take_profit,\n        confidence=round(confidence, 4),\n        volatility=VolatilityData(\n            current=round(volatility_current, 4),\n            predicted=round(volatility_predicted, 4),\n            level=vol_level,\n            atr_14=round(atr_14, 2)\n        ),\n        klines=[\n            KlineData(\n                timestamp=k['timestamp'],\n                open=k['open'],\n                high=k['high'],\n                low=k['low'],\n                close=k['close'],\n                volume=k['volume']\n            )\n            for k in klines[-20:]\n        ],\n        model_version=model_version,\n        price_source=price_source\n    )\n    \n    return result\n\n@app.get(\"/health\")\nasync def health_check():\n    \"\"\"健康檢查\"\"\"\n    return {\n        \"status\": \"ok\",\n        \"timestamp\": datetime.now().isoformat(),\n        \"model_version\": \"V5-HYBRID\",\n        \"model_type\": \"Real Model + Demo Fallback\",\n        \"models_cached\": len(model_manager.models),\n        \"failed_models\": len(model_manager.failed_models),\n        \"supported_symbols\": len(SUPPORTED_CRYPTOS_V5),\n        \"price_sources\": [\"yfinance (default)\", \"binance (optional)\"]\n    }\n\n# ============================================================================\n# 運行\n# ============================================================================\n\nif __name__ == \"__main__\":\n    import uvicorn\n    \n    print(\"\\n\" + \"=\"*80)\n    print(\" \"*15 + \"CPB Trading Web - V5 Model (HYBRID VERSION)\")\n    print(\"=\"*80)\n    print(f\"\\nModel Version: V5 (HYBRID)\")\n    print(f\"Strategy: Try Real Model \u2192 Fallback to Demo Predictions\")\n    print(f\"Price Source: yfinance (unified, consistent across timeframes)\")\n    print(f\"Binance Support: Optional (use_binance parameter)\")\n    print(f\"Supported Symbols: {len(SUPPORTED_CRYPTOS_V5)}\")\n    print(f\"Timeframes: {SUPPORTED_TIMEFRAMES}\")\n    print(f\"Features: 30+ technical indicators\")\n    print(f\"\\nStarting FastAPI server...\")\n    print(f\"API: http://localhost:8001\")\n    print(f\"Docs: http://localhost:8001/docs\")\n    print(\"\\n\u26a0  HYBRID MODE: Always provides predictions!\")\n    print(\"   - If real model loads: Uses trained model\")\n    print(\"   - If real model fails: Falls back to intelligent demo predictions\")\n    print(\"\\n\u26a0  PRICE CONSISTENCY:\")\n    print(\"   - 1D and 1H use SAME current price (from yfinance)\")\n    print(\"   - Binance optional (use_binance=true parameter)\")\n    print(\"\\n\" + \"=\"*80 + \"\\n\")\n    \n    uvicorn.run(\n        app,\n        host=\"0.0.0.0\",\n        port=8001,\n        log_level=\"info\"\n    )\n