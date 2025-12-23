#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - FastAPI Backend
實時抓取幣種資料 + 模型預測 + 開單點位推薦
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import torch
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import ccxt
import asyncio
from datetime import datetime
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI 初始化
# ============================================================================
app = FastAPI(title="CPB Trading Prediction API", version="1.0.0")

# CORS
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
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = os.environ.get('HF_USERNAME', 'zongowo111')
REPO_ID = f"{HF_USERNAME}/cpbmodel"

MODEL_CACHE_DIR = Path('./models_cache')
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# 支持的幣種
SUPPORTED_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'LINKUSDT', 'UNIUSDT',
    'AVAXUSDT', 'ATOMUSDT', 'VETUSDT', 'GRTUSDT', 'AXSUSDT',
    'BCHUSDT', 'MANAUSDT', 'SANDUSDT', 'XLMUSDT'
]

# ============================================================================
# 資料模型
# ============================================================================
class PredictionRequest(BaseModel):
    coin: str  # 例: BTCUSDT
    lookback_periods: int = 20  # 過去多少根K棒作為輸入
    prediction_horizon: int = 5  # 預測後多少根K棒

class KlineData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictionResult(BaseModel):
    coin: str
    timestamp: str
    current_price: float
    predicted_price_3: float  # 3根K棒後預測價格
    predicted_price_5: float  # 5根K棒後預測價格
    recommendation: str  # BUY / SELL / HOLD
    entry_price: float  # 建議開單價
    stop_loss: float  # 止損點
    take_profit: float  # 止盈點
    confidence: float  # 信心指數 (0-1)
    klines: List[KlineData]  # 用於預測的K棒數據

# ============================================================================
# 模型載入
# ============================================================================
class ModelManager:
    def __init__(self):
        self.models = {}  # {coin: model}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, coin: str):
        """從 HF 載入模型"""
        if coin in self.models:
            return self.models[coin]
        
        model_name = f"{coin}_1h_v1"
        logger.info(f"Loading model: {model_name}")
        
        try:
            if not HF_TOKEN:
                logger.warning("HF_TOKEN not set, using demo mode")
                # Demo 模式
                self.models[coin] = {'demo': True}
                return self.models[coin]
            
            # 從 HF 下載 model.bin
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{model_name}/pytorch_model.bin",
                cache_dir=str(MODEL_CACHE_DIR),
                token=HF_TOKEN
            )
            
            # 簡單的線性模型（假設保存的是 PyTorch checkpoint）
            model_data = torch.load(model_path, map_location=self.device)
            self.models[coin] = model_data
            logger.info(f"Model {coin} loaded successfully")
            return model_data
        
        except Exception as e:
            logger.error(f"Failed to load model {coin}: {e}")
            logger.warning(f"Using demo/fallback mode for {coin}")
            self.models[coin] = {'demo': True}
            return self.models[coin]
    
    def predict(self, coin: str, klines: np.ndarray) -> Dict:
        """執行預測
        
        Args:
            coin: 幣種
            klines: shape (lookback, 4) - [open, high, low, close]
        
        Returns:
            prediction dict
        """
        model = self.load_model(coin)
        
        # 將 klines 轉為 tensor
        X = torch.FloatTensor(klines).to(self.device)
        X = X.unsqueeze(0)  # (1, lookback, 4)
        
        with torch.no_grad():
            # 假設模型輸出: [price_3, price_5, direction]
            if isinstance(model, dict) and model.get('demo'):
                # Demo 模式
                output = self._forward_simple_model(X, None)
            elif isinstance(model, dict) and 'model_state_dict' in model:
                # 這是完整的 checkpoint，需要重新構建模型
                output = self._forward_simple_model(X, model)
            else:
                output = model(X) if callable(model) else self._forward_simple_model(X, model)
        
        return output
    
    def _forward_simple_model(self, X: torch.Tensor, model_data: Optional[dict]) -> Dict:
        """簡單的推理邏輯（可根據實際模型調整）"""
        # 輸出 3 個值: [normalized_price_3, normalized_price_5, direction]
        batch_size = X.shape[0]
        
        # 取最後一根K棒的收盤價
        current_close = X[0, -1, 3].item()  # 最後一根K棒的收盤
        
        # 簡單趨勢預測
        recent_closes = X[0, :, 3].cpu().numpy()
        trend = np.mean(recent_closes[-5:]) - np.mean(recent_closes[:5])
        direction = 1 if trend > 0 else (-1 if trend < 0 else 0)  # 1:up, -1:down, 0:hold
        
        # 簡單預測: ±2% 和 ±3%
        pred_3 = current_close * (1 + 0.02 * direction)
        pred_5 = current_close * (1 + 0.03 * direction)
        
        return {
            'price_3': float(pred_3),
            'price_5': float(pred_5),
            'direction': int(direction),
            'confidence': 0.7
        }

model_manager = ModelManager()

# ============================================================================
# 實時資料獲取
# ============================================================================
class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()
    
    async def fetch_klines(self, coin: str, timeframe: str = '1h', limit: int = 20) -> List[Dict]:
        """從 Binance 取得 K 棒數據
        
        Args:
            coin: 幣種 (e.g., 'BTCUSDT')
            timeframe: 時間框架 (e.g., '1h')
            limit: 取回的根數
        
        Returns:
            List of klines with OHLCV data
        """
        try:
            logger.info(f"Fetching {limit} {timeframe} klines for {coin}")
            
            # 非同步調用（防止阻塞）
            loop = asyncio.get_event_loop()
            klines = await loop.run_in_executor(
                None,
                self.exchange.fetch_ohlcv,
                coin,
                timeframe,
                None,
                limit
            )
            
            # 轉換格式
            result = []
            for k in klines:
                result.append({
                    'timestamp': k[0],
                    'open': k[1],
                    'high': k[2],
                    'low': k[3],
                    'close': k[4],
                    'volume': k[5]
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to fetch klines for {coin}: {e}")
            raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")
    
    def normalize_klines(self, klines: List[Dict]) -> np.ndarray:
        """正規化 K 棒數據
        
        Returns:
            shape (len(klines), 4) - [open, high, low, close]
        """
        data = []
        for k in klines:
            data.append([
                k['open'],
                k['high'],
                k['low'],
                k['close']
            ])
        
        data = np.array(data, dtype=np.float32)
        
        # Min-Max 正規化
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        data_norm = (data - min_val) / (max_val - min_val + 1e-8)
        
        return data_norm

data_fetcher = DataFetcher()

# ============================================================================
# API 端點
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "CPB Trading Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/coins": "List supported coins",
            "/predict": "Get prediction for a coin",
            "/predict-batch": "Batch predict multiple coins",
            "/health": "Health check"
        }
    }

@app.get("/coins")
async def get_coins():
    """列出所有支持的幣種"""
    return {
        "coins": SUPPORTED_COINS,
        "total": len(SUPPORTED_COINS)
    }

@app.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResult:
    """預測交易信號和開單點位"""
    
    # 驗證幣種
    if request.coin not in SUPPORTED_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Coin {request.coin} not supported. Supported: {SUPPORTED_COINS}"
        )
    
    # 1. 獲取實時 K 棒數據
    klines = await data_fetcher.fetch_klines(
        request.coin,
        timeframe='1h',
        limit=request.lookback_periods
    )
    
    current_price = klines[-1]['close']
    logger.info(f"{request.coin} current price: {current_price}")
    
    # 2. 正規化數據
    normalized_data = data_fetcher.normalize_klines(klines)
    
    # 3. 執行預測
    pred_result = model_manager.predict(request.coin, normalized_data)
    
    # 4. 計算開單點位
    price_3 = pred_result['price_3']
    price_5 = pred_result['price_5']
    direction = pred_result['direction']
    confidence = pred_result['confidence']
    
    # 推薦點位
    if direction > 0:  # 看漲
        recommendation = "BUY"
        entry_price = current_price  # 當前價格開單
        stop_loss = current_price * 0.98  # 止損 2%
        take_profit = price_5 * 1.02 if price_5 > current_price else price_5
    elif direction < 0:  # 看跌
        recommendation = "SELL"
        entry_price = current_price
        stop_loss = current_price * 1.02  # 止損 2%
        take_profit = price_5 * 0.98 if price_5 < current_price else price_5
    else:  # 保持
        recommendation = "HOLD"
        entry_price = current_price
        stop_loss = current_price * 0.99
        take_profit = current_price * 1.01
    
    # 5. 構建回應
    result = PredictionResult(
        coin=request.coin,
        timestamp=datetime.now().isoformat(),
        current_price=current_price,
        predicted_price_3=price_3,
        predicted_price_5=price_5,
        recommendation=recommendation,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        klines=[
            KlineData(
                time=datetime.fromtimestamp(k['timestamp']/1000).isoformat(),
                open=k['open'],
                high=k['high'],
                low=k['low'],
                close=k['close'],
                volume=k['volume']
            )
            for k in klines
        ]
    )
    
    return result

@app.post("/predict-batch")
async def predict_batch(coins: List[str]) -> List[PredictionResult]:
    """批量預測多個幣種"""
    results = []
    for coin in coins:
        try:
            result = await predict(PredictionRequest(coin=coin))
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to predict {coin}: {e}")
            continue
    return results

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "device": str(model_manager.device),
        "models_cached": len(model_manager.models)
    }

# ============================================================================
# 運行
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
