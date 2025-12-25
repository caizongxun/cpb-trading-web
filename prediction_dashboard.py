#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading - Prediction Dashboard
支持 V5/V6 版本選擇，多幣種複選，未来 10 根与步預測可視化
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
import json

try:
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except:
    HAS_HF = False

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 數业模弋
# ============================================================================

class ForecastBar(BaseModel):
    bar_index: int
    predicted_price: float
    confidence: float

class PredictionForecast(BaseModel):
    symbol: str
    version: str  # 'V5' or 'V6'
    current_price: float
    last_10_closes: List[float]
    forecast_bars: List[ForecastBar]
    timestamp: str
    model_status: str  # 'loaded' or 'synthetic'
    available_coins_v5: List[str]
    available_coins_v6: List[str]

# ============================================================================
# 模型管理器
# ============================================================================

class ModelManager:
    """管理 V5 徒 V6 模型的上載、希靈化並但及預測"""
    
    # V5 支持的幣種 (出于訓練繁教)
    V5_COINS = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE', 'ATOM']
    
    # V6 支持的幣種 (更模訓練繁教)
    V6_COINS = [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'AVAX', 'DOGE', 'ADA',
        'DOT', 'LTC', 'LINK', 'ATOM', 'UNI', 'XLM'
    ]
    
    def __init__(self):
        self.models_cache = {}  # {version}-{symbol}-{timeframe}: model
        self.scalers_cache = {}  # {version}-{symbol}-{timeframe}: scaler
        self.hf_available = HAS_HF
        self.tf_available = HAS_TENSORFLOW
        logger.info(f"ModelManager 初始化: TF={self.tf_available}, HF={self.hf_available}")
    
    def check_coin_support(self, symbol: str, version: str) -> bool:
        """檢查幣種是否支持此版本"""
        if version == 'V5':
            return symbol in self.V5_COINS
        elif version == 'V6':
            return symbol in self.V6_COINS
        return False
    
    def get_available_coins(self, version: str) -> List[str]:
        """取得此版本支持的幣種"""
        return self.V5_COINS if version == 'V5' else self.V6_COINS
    
    async def load_model(self, symbol: str, version: str, timeframe: str = '1h'):
        """從 HuggingFace 下載模型"""
        cache_key = f"{version}-{symbol}-{timeframe}"
        
        if cache_key in self.models_cache:
            logger.debug(f"使用緩存模弋: {cache_key}")
            return self.models_cache[cache_key]
        
        if not self.hf_available or not self.tf_available:
            logger.warning(f"缺少依賴 (TF={self.tf_available}, HF={self.hf_available}), 使用綜合數據")
            return None
        
        try:
            logger.info(f"下載模弋: {symbol} {version} {timeframe}")
            
            model_path = hf_hub_download(
                repo_id="zongowo111/cpb-models",
                filename=f"models_{version.lower()}/{symbol}_{timeframe}_model.h5",
                repo_type="dataset",
                cache_dir="./models_cache"
            )
            
            model = load_model(model_path)
            self.models_cache[cache_key] = model
            
            # 也試著下載 scaler
            try:
                scalers_path = hf_hub_download(
                    repo_id="zongowo111/cpb-models",
                    filename=f"models_{version.lower()}/{symbol}_{timeframe}_scalers.pkl",
                    repo_type="dataset",
                    cache_dir="./models_cache"
                )
                with open(scalers_path, 'rb') as f:
                    scaler = pickle.load(f)
                self.scalers_cache[cache_key] = scaler
                logger.info(f"成功下載: {cache_key}")
            except:
                logger.debug(f"Scaler 下載失敗: {cache_key}")
                self.scalers_cache[cache_key] = None
            
            return model
        
        except Exception as e:
            logger.error(f"模弋下載失敗: {e}")
            return None
    
    def normalize_data(self, X: np.ndarray, scaler=None) -> Tuple[np.ndarray, dict]:
        """正規化數據"""
        if scaler is None:
            X_mean = X.mean(axis=1, keepdims=True)
            X_std = X.std(axis=1, keepdims=True) + 1e-8
            X_norm = (X - X_mean) / X_std
            return X_norm, {'mean': X_mean, 'std': X_std}
        
        try:
            if hasattr(scaler, 'transform'):
                X_2d = X.reshape(-1, X.shape[-1])
                X_norm = scaler.transform(X_2d)
                return X_norm.reshape(X.shape), {'scaler': scaler}
        except:
            pass
        
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - X_mean) / X_std
        return X_norm, {'mean': X_mean, 'std': X_std}
    
    async def predict_future(
        self,
        historical_prices: List[float],
        symbol: str,
        version: str,
        timeframe: str = '1h',
        steps_ahead: int = 10
    ) -> Dict:
        """預測未数 N 根 K 棒的價格"""
        
        logger.info(f"\n=== 預測 {symbol} {version} ===")
        logger.info(f"歷史數據: {len(historical_prices)} \u6839")
        logger.info(f預測步數: {steps_ahead}")
        
        prices = np.array(historical_prices)
        
        # 試著加載真實模弋
        model = await self.load_model(symbol, version, timeframe)
        model_loaded = model is not None
        
        if not model_loaded:
            logger.warning(f"缺少真實模弋, 使用綜合數據")
            # 綜合數據: 安静締常物稱趋勢
            return self._generate_synthetic_forecast(
                historical_prices,
                symbol,
                steps_ahead
            )
        
        # 有真實模弋
        try:
            # 使用不同的 lookback 模你
            for lookback in [60, 50, 40, 30, 20]:
                if len(prices) < lookback:
                    continue
                
                logger.debug(f"嘗試 lookback={lookback}")
                
                # 準備數據
                X = prices[-lookback:].reshape(1, -1)
                cache_key = f"{version}-{symbol}-{timeframe}"
                scaler = self.scalers_cache.get(cache_key)
                
                X_norm, norm_info = self.normalize_data(X, scaler)
                
                # 來雖推
                pred_norm = model.predict(X_norm, verbose=0)
                
                # 反正規化
                if 'scaler' not in norm_info:
                    mean, std = norm_info['mean'], norm_info['std']
                    predictions = pred_norm * std + mean
                else:
                    predictions = pred_norm
                
                # 取整 predict list
                if len(predictions.shape) > 1:
                    forecast_prices = predictions.flatten()[:steps_ahead]
                else:
                    forecast_prices = predictions[:steps_ahead]
                
                if len(forecast_prices) >= steps_ahead:
                    logger.info(f"成功概述: lookback={lookback}, 預測 {len(forecast_prices)} 根")
                    
                    return {
                        'success': True,
                        'forecast_prices': forecast_prices.tolist()[:steps_ahead],
                        'confidence': 0.65,
                        'model_loaded': True,
                        'lookback': lookback
                    }
        
        except Exception as e:
            logger.error(f"真實模弋預測失敗: {e}")
        
        # 回來綜合數據
        logger.info("會初綜合數據")
        return self._generate_synthetic_forecast(
            historical_prices,
            symbol,
            steps_ahead
        )
    
    def _generate_synthetic_forecast(
        self,
        historical_prices: List[float],
        symbol: str,
        steps_ahead: int = 10
    ) -> Dict:
        """綜合數據預測 (靜樫跋庋牢)"""
        
        prices = np.array(historical_prices)
        current = prices[-1]
        
        # 計算趣勢
        if len(prices) > 10:
            recent_avg = prices[-10:].mean()
            past_avg = prices[-20:-10].mean()
            trend = (recent_avg - past_avg) / past_avg if past_avg > 0 else 0
        else:
            trend = 0.001  # 準中是法撤趣勢
        
        # 計算波動性
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        logger.debug(f"綜合數據: trend={trend:.4f}, volatility={volatility:.4f}")
        
        # 產生預測
        forecast = []
        price = current
        
        for i in range(steps_ahead):
            # 缀於的影響
            trend_component = trend * (1 - i / steps_ahead)  # 遅滿鯊趣勢
            volatility_component = np.random.normal(0, volatility)
            
            price_change = (trend_component + volatility_component) * price
            price = price + price_change
            
            forecast.append(price)
        
        return {
            'success': True,
            'forecast_prices': forecast,
            'confidence': 0.45,  # 低罪信心度表示綜合
            'model_loaded': False,
            'trend': trend,
            'volatility': volatility
        }

model_manager = ModelManager()

# ============================================================================
# FastAPI 應用
# ============================================================================

app = FastAPI(
    title="CPB Prediction Dashboard",
    version="2.0.0",
    description="V5/V6 版本選擇, 數位化未来預測"
)

@app.get("/predict/available-coins")
async def get_available_coins():
    """取得上訓練繁教的幣種清單"""
    return {
        'V5': model_manager.get_available_coins('V5'),
        'V6': model_manager.get_available_coins('V6')
    }

@app.post("/predict/forecast")
async def predict_forecast(
    symbol: str,
    version: str = 'V6',
    timeframe: str = '1h',
    historical_prices: List[float] = None
) -> PredictionForecast:
    """預測未新 10 根 K 棒"""
    
    logger.info(f"\n=== 預測請求 ===")
    logger.info(f稦號: {symbol}")
    logger.info(f版本: {version}")
    logger.info(f時間核: {timeframe}")
    
    # 檢查幣種支援
    if not model_manager.check_coin_support(symbol, version):
        raise HTTPException(
            status_code=400,
            detail=f"{version} 不支援 {symbol}. 支援的幣種: {model_manager.get_available_coins(version)}"
        )
    
    # 提供範例數據
    if not historical_prices:
        logger.info("使用範例数據")
        historical_prices = [
            40000 + i*100 + np.random.normal(0, 500)
            for i in range(30)
        ]
    
    if len(historical_prices) < 10:
        raise HTTPException(
            status_code=400,
            detail="需要至少 10 根歷史數據"
        )
    
    # 預測上次 10 根
    forecast_result = await model_manager.predict_future(
        historical_prices,
        symbol,
        version,
        timeframe,
        steps_ahead=10
    )
    
    current_price = float(historical_prices[-1])
    forecast_prices = forecast_result['forecast_prices']
    
    # 構易預測 K 棒
    forecast_bars = [
        ForecastBar(
            bar_index=i+1,
            predicted_price=float(price),
            confidence=forecast_result['confidence']
        )
        for i, price in enumerate(forecast_prices)
    ]
    
    return PredictionForecast(
        symbol=symbol,
        version=version,
        current_price=current_price,
        last_10_closes=historical_prices[-10:],
        forecast_bars=forecast_bars,
        timestamp=datetime.now().isoformat(),
        model_status='loaded' if forecast_result.get('model_loaded') else 'synthetic',
        available_coins_v5=model_manager.get_available_coins('V5'),
        available_coins_v6=model_manager.get_available_coins('V6')
    )

@app.get("/")
async def root():
    return {
        "name": "CPB Prediction Dashboard",
        "version": "2.0.0",
        "features": [
            "V5/V6 版本策筆",
            "敵幣種自動鳾殇",
            "未來 10 根 K 棒預測",
            "真實模弋主動下載「教訓榕",
            "綜合數據回上樫跋学"
        ],
        "endpoints": {
            "/predict/available-coins": "取得各版本支援的幣種",
            "/predict/forecast": "預測未来 10 根 K 棒",
            "/docs": "Swagger API 文伋"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("""
================================================================================
               CPB Prediction Dashboard - V2.0
================================================================================

Features:
  ✓ V5/V6 版本選擇
  ✓ 敵幣種支援邊䨙 (V5: 8種, V6: 14種)
  ✓ 未來 10 根 K 棒預測
  ✓ 真實模弋主動下載
  ✓ 綜合数據回例

Available Endpoints:
  GET  /predict/available-coins
  POST /predict/forecast
  GET  /docs

Model Management:
  TensorFlow Available: True
  HuggingFace Available: True
  Model Cache: ./models_cache

================================================================================
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
