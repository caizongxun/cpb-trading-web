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
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import asyncio
from datetime import datetime
import logging
import math

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

# 支援的幣種
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

class VolatilityData(BaseModel):
    current: float  # 當前波動率 (%) - 最後一根K線的價格變化
    predicted_3: float  # 3根K棒後預測波動率 (%) - 預測價格變化
    predicted_5: float  # 5根K棒後預測波動率 (%) - 預測價格變化
    volatility_level: str  # "低" / "中" / "高"
    atr_14: float  # 14根K棒平均真實幅度

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
    volatility: VolatilityData  # 波動率資料
    klines: List[KlineData]  # 用於預測的K棒數據

# ============================================================================
# 模型管理
# ============================================================================
class ModelManager:
    def __init__(self):
        self.models = {}  # {coin: model}
        logger.info("ModelManager initialized (demo mode)")
    
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
            
            logger.info(f"Model {coin} loaded successfully")
            self.models[coin] = {'path': model_path, 'demo': False}
            return self.models[coin]
        
        except Exception as e:
            logger.warning(f"Failed to load model {coin}: {e}")
            logger.warning(f"Using demo/fallback mode for {coin}")
            self.models[coin] = {'demo': True}
            return self.models[coin]
    
    def predict(self, coin: str, klines: List[Dict]) -> Dict:
        """執行預測
        
        Args:
            coin: 幣種
            klines: K棒數據列表
        
        Returns:
            prediction dict
        """
        model = self.load_model(coin)
        
        # 簡單的 Demo 預測邏輯（不使用 PyTorch）
        if model.get('demo'):
            return self._forward_simple_model(klines)
        else:
            return self._forward_simple_model(klines)
    
    def _forward_simple_model(self, klines: List[Dict]) -> Dict:
        """簡單的推理邏輯（不依賴 PyTorch/NumPy）"""
        # 提取收盤價
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        opens = [k['open'] for k in klines]
        current_close = closes[-1]
        
        # === 計算波動率 (價格變化百分比) ===
        # 當前波動率 = 最後一根K線的 (close - open) / open * 100%
        last_kline = klines[-1]
        volatility_current = ((last_kline['close'] - last_kline['open']) / last_kline['open']) * 100
        
        # 計算平均真實幅度 (ATR)
        true_ranges = []
        for i in range(len(klines)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1] if i > 0 else low
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        atr_14 = sum(true_ranges[-14:]) / min(14, len(true_ranges)) if true_ranges else 0
        
        # 趨勢判斷
        recent_avg = sum(closes[-5:]) / 5
        past_avg = sum(closes[:5]) / 5
        trend = recent_avg - past_avg
        
        if trend > 0:
            direction = 1  # 看漲
        elif trend < 0:
            direction = -1  # 看跌
        else:
            direction = 0  # 持平
        
        # 預測價格
        pred_3 = current_close * (1 + 0.02 * direction)  # ±2%
        pred_5 = current_close * (1 + 0.03 * direction)  # ±3%
        
        # 預測波動率 (基於預測價格變化)
        volatility_pred_3 = ((pred_3 - current_close) / current_close) * 100
        volatility_pred_5 = ((pred_5 - current_close) / current_close) * 100
        
        # 波動率等級 (基於最後K線的變化幅度)
        abs_volatility = abs(volatility_current)
        if abs_volatility < 0.5:
            vol_level = "低"
        elif abs_volatility < 1.5:
            vol_level = "中"
        else:
            vol_level = "高"
        
        return {
            'price_3': round(pred_3, 2),
            'price_5': round(pred_5, 2),
            'direction': direction,
            'confidence': 0.65,
            'volatility_current': volatility_current,  # 直接返回百分比
            'volatility_pred_3': volatility_pred_3,    # 直接返回百分比
            'volatility_pred_5': volatility_pred_5,    # 直接返回百分比
            'volatility_level': vol_level,
            'atr_14': round(atr_14, 2),
        }

model_manager = ModelManager()

# ============================================================================
# 實時資料獲取
# ============================================================================
class DataFetcher:
    def __init__(self):
        try:
            import ccxt
            self.exchange = ccxt.binance()
            self.available = True
        except Exception as e:
            logger.warning(f"CCXT not available: {e}")
            self.available = False
            self.exchange = None
    
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
            if not self.available:
                logger.warning(f"CCXT not available, returning demo data for {coin}")
                return self._generate_demo_klines(limit)
            
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
            logger.info(f"Falling back to demo data")
            return self._generate_demo_klines(limit)
    
    def _generate_demo_klines(self, limit: int = 20) -> List[Dict]:
        """生成演示 K 棒數據"""
        import time
        import random
        
        klines = []
        base_price = 42000  # 假設基準價
        current_time = int(time.time() * 1000) - (limit * 3600 * 1000)
        
        for i in range(limit):
            price_change = random.uniform(-100, 100)
            open_price = base_price + price_change
            close_price = open_price + random.uniform(-50, 50)
            high_price = max(open_price, close_price) + random.uniform(0, 100)
            low_price = min(open_price, close_price) - random.uniform(0, 100)
            
            klines.append({
                'timestamp': current_time + (i * 3600 * 1000),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.uniform(100, 1000)
            })
            
            base_price = close_price
        
        return klines

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
    """列出所有支援的幣種"""
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
    
    # 2. 執行預測
    pred_result = model_manager.predict(request.coin, klines)
    
    # 3. 計算開單點位
    price_3 = pred_result['price_3']
    price_5 = pred_result['price_5']
    direction = pred_result['direction']
    confidence = pred_result['confidence']
    volatility_current = pred_result['volatility_current']
    volatility_pred_3 = pred_result['volatility_pred_3']
    volatility_pred_5 = pred_result['volatility_pred_5']
    volatility_level = pred_result['volatility_level']
    atr_14 = pred_result['atr_14']
    
    # 推薦點位
    if direction > 0:  # 看漲
        recommendation = "BUY"
        entry_price = current_price
        stop_loss = round(current_price * 0.98, 2)  # 止損 2%
        take_profit = round(price_5 * 1.02, 2) if price_5 > current_price else round(price_5, 2)
    elif direction < 0:  # 看跌
        recommendation = "SELL"
        entry_price = current_price
        stop_loss = round(current_price * 1.02, 2)  # 止損 2%
        take_profit = round(price_5 * 0.98, 2) if price_5 < current_price else round(price_5, 2)
    else:  # 持有
        recommendation = "HOLD"
        entry_price = current_price
        stop_loss = round(current_price * 0.99, 2)
        take_profit = round(current_price * 1.01, 2)
    
    # 4. 構建回應
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
        volatility=VolatilityData(
            current=round(volatility_current, 2),  # 已經是百分比
            predicted_3=round(volatility_pred_3, 2),
            predicted_5=round(volatility_pred_5, 2),
            volatility_level=volatility_level,
            atr_14=atr_14
        ),
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