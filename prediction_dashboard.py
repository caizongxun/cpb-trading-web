#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading - Prediction Dashboard
支持 V5/V6 版本選擇，多幣種複選，未來 10 根 K 棒預測可視化
動態棂探實際上有的模型，而不是依賴預訮的幣種清單
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
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
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF = True
except:
    HAS_HF = False

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 數據模型
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
    """管理 V5 和 V6 模型的上載、初始化並執行預測"""
    
    # 預訮的幣種 (如果沒有指定其他的候選)
    FALLBACK_V5_COINS = [
        'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOGE', 'ATOM',
        'DOT', 'LTC', 'LINK', 'UNI', 'AVAX', 'XLM'
    ]
    
    FALLBACK_V6_COINS = [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'AVAX', 'DOGE', 'ADA',
        'DOT', 'LTC', 'LINK', 'ATOM', 'UNI', 'XLM', 'MATIC', 'ARB'
    ]
    
    def __init__(self):
        self.models_cache = {}  # {version}-{symbol}-{timeframe}: model
        self.scalers_cache = {}  # {version}-{symbol}-{timeframe}: scaler
        self.available_coins_cache = {}  # {version}: [coins]
        self.hf_available = HAS_HF
        self.tf_available = HAS_TENSORFLOW
        
        logger.info(f"ModelManager 初始化: TF={self.tf_available}, HF={self.hf_available}")
        
        # 立即棂探可用的幣種
        self._discover_available_coins()
    
    def _discover_available_coins(self):
        """從 HuggingFace 模型倉庫棂探實際可用的幣種"""
        logger.info("開始棂探 HuggingFace 模型倉庫中的幣種...")
        
        for version in ['V5', 'V6']:
            coins: Set[str] = set()
            
            if not self.hf_available:
                logger.warning(f"{version}: HuggingFace 不可用，使用預訮值")
                coins = set(self.FALLBACK_V5_COINS if version == 'V5' else self.FALLBACK_V6_COINS)
            else:
                try:
                    logger.info(f"{version}: 正在查詢 HuggingFace...")
                    
                    # 列出模型倉庫中的所有模型檔
                    files = list_repo_files(
                        repo_id="zongowo111/cpb-models",
                        repo_type="dataset"
                    )
                    
                    # 歷遍檔案，找出此版本的模型
                    for file_path in files:
                        # 例如: models_v5/BTC_1h_model.h5 或 models_v6/ETH_1d_model.h5
                        if f'models_{version.lower()}/' in file_path and '_model.h5' in file_path:
                            # 拆解檔案名
                            parts = file_path.split('/')[-1]  # 取最後一部分
                            symbol = parts.split('_')[0]  # 取 BTC 或 ETH
                            coins.add(symbol)
                    
                    if coins:
                        logger.info(f"{version}: 找到 {len(coins)} 個幣種: {sorted(coins)}")
                    else:
                        logger.warning(f"{version}: 沒有找到模型，使用預訮值")
                        coins = set(self.FALLBACK_V5_COINS if version == 'V5' else self.FALLBACK_V6_COINS)
                
                except Exception as e:
                    logger.error(f"{version}: 棂探失敗 - {e}，使用預訮值")
                    coins = set(self.FALLBACK_V5_COINS if version == 'V5' else self.FALLBACK_V6_COINS)
            
            # 存储排序的幣種
            self.available_coins_cache[version] = sorted(list(coins))
        
        logger.info(f"幣種棂探完成")
        logger.info(f"V5 ({len(self.available_coins_cache['V5'])} 種): {self.available_coins_cache['V5']}")
        logger.info(f"V6 ({len(self.available_coins_cache['V6'])} 種): {self.available_coins_cache['V6']}")
    
    def check_coin_support(self, symbol: str, version: str) -> bool:
        """檢查幣種是否支持此版本"""
        available = self.available_coins_cache.get(version, [])
        return symbol in available
    
    def get_available_coins(self, version: str) -> List[str]:
        """取得此版本支持的幣種"""
        return self.available_coins_cache.get(version, [])
    
    async def load_model(self, symbol: str, version: str, timeframe: str = '1h'):
        """從 HuggingFace 下載模型"""
        cache_key = f"{version}-{symbol}-{timeframe}"
        
        if cache_key in self.models_cache:
            logger.debug(f"使用緩存模型: {cache_key}")
            return self.models_cache[cache_key]
        
        if not self.hf_available or not self.tf_available:
            logger.warning(f"缺少依賴 (TF={self.tf_available}, HF={self.hf_available}), 使用綜合數據")
            return None
        
        try:
            logger.info(f"下載模型: {symbol} {version} {timeframe}")
            
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
            logger.error(f"模型下載失敗: {e}")
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
        """預測未來 N 根 K 棒的價格"""
        
        logger.info(f"\n=== 預測 {symbol} {version} ===")
        logger.info(f"歷史數據: {len(historical_prices)} 根")
        logger.info(f"預測步數: {steps_ahead}")
        
        prices = np.array(historical_prices)
        
        # 試著加載真實模型
        model = await self.load_model(symbol, version, timeframe)
        model_loaded = model is not None
        
        if not model_loaded:
            logger.warning(f"缺少真實模型, 使用綜合數據")
            return self._generate_synthetic_forecast(
                historical_prices,
                symbol,
                steps_ahead
            )
        
        # 有真實模型
        try:
            # 使用不同的 lookback 模式
            for lookback in [60, 50, 40, 30, 20]:
                if len(prices) < lookback:
                    continue
                
                logger.debug(f"嘗試 lookback={lookback}")
                
                # 準備數據
                X = prices[-lookback:].reshape(1, -1)
                cache_key = f"{version}-{symbol}-{timeframe}"
                scaler = self.scalers_cache.get(cache_key)
                
                X_norm, norm_info = self.normalize_data(X, scaler)
                
                # 執行推論
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
                    logger.info(f"成功預測: lookback={lookback}, 預測 {len(forecast_prices)} 根")
                    
                    return {
                        'success': True,
                        'forecast_prices': forecast_prices.tolist()[:steps_ahead],
                        'confidence': 0.65,
                        'model_loaded': True,
                        'lookback': lookback
                    }
        
        except Exception as e:
            logger.error(f"真實模型預測失敗: {e}")
        
        # 回到綜合數據
        logger.info("使用綜合數據")
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
        """綜合數據預測 (靜態基礎模式)"""
        
        prices = np.array(historical_prices)
        current = prices[-1]
        
        # 計算趨勢
        if len(prices) > 10:
            recent_avg = prices[-10:].mean()
            past_avg = prices[-20:-10].mean()
            trend = (recent_avg - past_avg) / past_avg if past_avg > 0 else 0
        else:
            trend = 0.001
        
        # 計算波動性
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        logger.debug(f"綜合數據: trend={trend:.4f}, volatility={volatility:.4f}")
        
        # 產生預測
        forecast = []
        price = current
        
        for i in range(steps_ahead):
            # 趨勢的影響
            trend_component = trend * (1 - i / steps_ahead)
            volatility_component = np.random.normal(0, volatility)
            
            price_change = (trend_component + volatility_component) * price
            price = price + price_change
            
            forecast.append(price)
        
        return {
            'success': True,
            'forecast_prices': forecast,
            'confidence': 0.45,
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
    description="V5/V6 版本選擇, 數位化未來預測"
)

@app.get("/predict/available-coins")
async def get_available_coins():
    """取得各訓練版本的幣種清單"""
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
    """預測未來 10 根 K 棒"""
    
    logger.info(f"\n=== 預測請求 ===")
    logger.info(f"幣號: {symbol}")
    logger.info(f"版本: {version}")
    logger.info(f"時間核: {timeframe}")
    
    # 檢查幣種支援
    if not model_manager.check_coin_support(symbol, version):
        raise HTTPException(
            status_code=400,
            detail=f"{version} 不支援 {symbol}. 支援的幣種: {model_manager.get_available_coins(version)}"
        )
    
    # 提供範例數據
    if not historical_prices:
        logger.info("使用範例數據")
        historical_prices = [
            40000 + i*100 + np.random.normal(0, 500)
            for i in range(30)
        ]
    
    if len(historical_prices) < 10:
        raise HTTPException(
            status_code=400,
            detail="需要至少 10 根歷史數據"
        )
    
    # 預測未來 10 根
    forecast_result = await model_manager.predict_future(
        historical_prices,
        symbol,
        version,
        timeframe,
        steps_ahead=10
    )
    
    current_price = float(historical_prices[-1])
    forecast_prices = forecast_result['forecast_prices']
    
    # 構築預測 K 棒
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
    v5_coins = model_manager.get_available_coins('V5')
    v6_coins = model_manager.get_available_coins('V6')
    
    return {
        "name": "CPB Prediction Dashboard",
        "version": "2.0.0",
        "features": [
            "V5/V6 版本選擇",
            "多幣種自動棂探",
            "未來 10 根 K 棒預測",
            "真實模型自動下載 (HuggingFace)",
            "綜合數據回退機制"
        ],
        "coin_counts": {
            "V5": len(v5_coins),
            "V6": len(v6_coins)
        },
        "available_coins": {
            "V5": v5_coins,
            "V6": v6_coins
        },
        "endpoints": {
            "/predict/available-coins": "取得各版本支援的幣種",
            "/predict/forecast": "預測未來 10 根 K 棒",
            "/docs": "Swagger API 文檔"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    v5_coins = model_manager.get_available_coins('V5')
    v6_coins = model_manager.get_available_coins('V6')
    
    print("""
================================================================================
               CPB Prediction Dashboard - V2.1
================================================================================

Features:
  ✓ V5/V6 版本選擇
  ✓ 動態棂探 HuggingFace 中的幣種
  ✓ 未來 10 根 K 棒預測
  ✓ 真實模型自動下載
  ✓ 綜合數據回退

Available Endpoints:
  GET  /predict/available-coins
  POST /predict/forecast
  GET  /docs

Model Management:
  TensorFlow Available: {}
  HuggingFace Available: {}
  Model Cache: ./models_cache

Detected Coins:
  V5 ({}): {}
  V6 ({}): {}

================================================================================
    """.format(
        HAS_TENSORFLOW,
        HAS_HF,
        len(v5_coins),
        ', '.join(v5_coins),
        len(v6_coins),
        ', '.join(v6_coins)
    ))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
