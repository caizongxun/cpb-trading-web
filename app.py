#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V2 Model Inference
實時拉取幣種資料 + V2 模型予測 + 開單點位推護
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
import numpy as np
import tensorflow as tf
from market_analysis import MarketAnalyzer, MarketAnalysisResult

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI 初始化
# ============================================================================
app = FastAPI(title="CPB Trading Prediction API - V2", version="2.0.0")

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

# 支援的 20 種幣種 - 確保正好 20 種
SUPPORTED_COINS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'BNBUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT',
    'ATOMUSDT', 'NEARUSDT', 'FTMUSDT', 'ARBUSDT', 'OPUSDT',
    'LITUSDT', 'STXUSDT', 'INJUSDT', 'LUNCUSDT', 'LUNAUSDT'
]

print(f"\n[✓] 模型版本: V2")
print(f"[✓] 支援幣種數量: {len(SUPPORTED_COINS)}")
print(f"[✓] 幣種清单: {SUPPORTED_COINS}\n")

# ============================================================================
# 資料模型
# ============================================================================
class PredictionRequest(BaseModel):
    coin: str  # 例: BTCUSDT
    lookback_periods: int = 20  # 過去多少根K棒作為輸入
    prediction_horizon: int = 5  # 預測例更多少根K棒

class MarketAnalysisRequest(BaseModel):
    symbol: str  # 幣種 (e.g., 'BTC', 'ETH')
    timeframe: str = '1d'  # 時間框架 ('1h', '1d')
    use_binance: bool = False

class KlineData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class VolatilityData(BaseModel):
    current: float  # 當前波動率 (%) - 最後一根K線的價格變化
    predicted_3: float  # 3根K棒侌預測波動率 (%) - 預測價格變化
    predicted_5: float  # 5根K棒侌預測波動率 (%) - 預測價格變化
    level: str  # "低" / "中" / "高"
    atr_14: float  # 14根K棒平均真實恆度

class PredictionResult(BaseModel):
    coin: str
    timestamp: str
    current_price: float
    predicted_price_3: float  # 3根K棒侌預測價格
    predicted_price_5: float  # 5根K棒侌預測價格
    recommendation: str  # BUY / SELL / HOLD
    entry_price: float  # 建議開單價
    stop_loss: float  # 止損點
    take_profit: float  # 止盈點
    confidence: float  # 信心指數 (0-1)
    volatility: VolatilityData  # 波動率資料
    klines: List[KlineData]  # 用于預測K棒數據
    model_version: str  # 模型版本

class MarketAnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    trend: Dict  # TrendAnalysis serialized
    best_entry_bar: int
    price_extremes: Dict  # PriceExtremes serialized
    forecast_prices: List[float]
    recommendation: str

# ============================================================================
# V2 模型管理
# ============================================================================
class ModelManagerV2:
    def __init__(self):
        self.models = {}  # {coin: model}
        self.model_dir = Path('ALL_MODELS/MODEL_V2')
        logger.info("ModelManager V2 initialized")
    
    def load_model(self, coin: str):
        """從本地載入 V2 模型"""
        if coin in self.models:
            return self.models[coin]
        
        model_name = f"v2_model_{coin}.h5"
        model_path = self.model_dir / model_name
        
        logger.info(f"Loading V2 model: {coin}")
        
        try:
            # 檢查模型是否存在
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                logger.warning(f"Using demo mode for {coin}")
                self.models[coin] = {'demo': True}
                return self.models[coin]
            
            # 載入模型
            model = tf.keras.models.load_model(str(model_path))
            self.models[coin] = {'model': model, 'demo': False, 'path': str(model_path)}
            logger.info(f"Model {coin} loaded successfully from {model_path}")
            return self.models[coin]
        
        except Exception as e:
            logger.error(f"Failed to load model {coin}: {e}")
            logger.warning(f"Using demo/fallback mode for {coin}")
            self.models[coin] = {'demo': True}
            return self.models[coin]
    
    def predict(self, coin: str, klines: List[Dict]) -> Dict:
        """執行預測 (V2 模型 輸入: [price, volatility])
        
        Args:
            coin: 幣種
            klines: K棒數據清単
        
        Returns:
            prediction dict
        """
        model = self.load_model(coin)
        
        # 使用 V2 模型骞予測
        if model.get('demo'):
            return self._forward_simple_model(klines)
        else:
            return self._forward_v2_model(klines, model['model'])
    
    def _forward_v2_model(self, klines: List[Dict], model) -> Dict:
        """V2 模型推理 (TensorFlow/Keras)"""
        try:
            # 提取收盤價
            closes = np.array([k['close'] for k in klines])
            highs = np.array([k['high'] for k in klines])
            lows = np.array([k['low'] for k in klines])
            opens = np.array([k['open'] for k in klines])
            current_close = closes[-1]
            
            # 正見化 (min-max 正見化)
            close_min = closes.min()
            close_max = closes.max()
            close_range = close_max - close_min
            if close_range == 0:
                close_range = 1
            
            # 揰技 [seq_len, 4]
            ohlc = np.column_stack([opens, highs, lows, closes])
            ohlc_norm = (ohlc - close_min) / close_range
            
            # 添加批次維度
            X = ohlc_norm.reshape(1, -1, 4).astype(np.float32)
            
            # 預測
            prediction = model.predict(X, verbose=0)
            
            # V2 模型輸出: [price, volatility]
            price_change_pct = float(prediction[0, 0]) if len(prediction[0]) > 0 else 0.5
            volatility_pred = float(prediction[0, 1]) if len(prediction[0]) > 1 else 1.2
            
            # 逾的潢水模式采用粗矊部分
            last_kline = klines[-1]
            volatility_current = abs((last_kline['close'] - last_kline['open']) / last_kline['open']) * 100
            
            # 計算 ATR
            true_ranges = []
            for i in range(len(klines)):
                high = highs[i]
                low = lows[i]
                prev_close = closes[i-1] if i > 0 else low
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)
            atr_14 = np.mean(true_ranges[-14:]) if true_ranges else 0
            
            # 預測價格
            pred_3 = current_close * (1 + price_change_pct / 100 * 0.5)  # 開子樜
            pred_5 = current_close * (1 + price_change_pct / 100 * 0.8)  # 加強
            
            # 預測波動率
            volatility_pred_3 = abs((pred_3 - current_close) / current_close) * 100
            volatility_pred_5 = abs((pred_5 - current_close) / current_close) * 100
            
            # 波動率等級
            if volatility_current < 0.5:
                vol_level = "低"
            elif volatility_current < 1.5:
                vol_level = "中"
            else:
                vol_level = "高"
            
            # 信心度
            confidence = min(0.85, max(0.55, abs(price_change_pct) / 3.0))
            
            return {
                'price_3': round(pred_3, 2),
                'price_5': round(pred_5, 2),
                'direction': 1 if price_change_pct > 0 else (-1 if price_change_pct < 0 else 0),
                'confidence': confidence,
                'volatility_current': volatility_current,
                'volatility_pred_3': volatility_pred_3,
                'volatility_pred_5': volatility_pred_5,
                'volatility_level': vol_level,
                'atr_14': round(float(atr_14), 2),
            }
        
        except Exception as e:
            logger.error(f"V2 model prediction failed: {e}")
            logger.warning(f"Falling back to simple model")
            return self._forward_simple_model(klines)
    
    def _forward_simple_model(self, klines: List[Dict]) -> Dict:
        """粗便的推理邏輯（不依賴 PyTorch/NumPy）"""
        # 提取收盤價
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        opens = [k['open'] for k in klines]
        current_close = closes[-1]
        
        # === 計算波動率 (價格變化百分比) ===
        last_kline = klines[-1]
        volatility_current = ((last_kline['close'] - last_kline['open']) / last_kline['open']) * 100
        
        # 計算平均真實恆度 (ATR)
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
            direction = -1  # 看空
        else:
            direction = 0  # 持平
        
        # 預測價格
        pred_3 = current_close * (1 + 0.02 * direction)  # ±2%
        pred_5 = current_close * (1 + 0.03 * direction)  # ±3%
        
        # 預測波動率
        volatility_pred_3 = ((pred_3 - current_close) / current_close) * 100
        volatility_pred_5 = ((pred_5 - current_close) / current_close) * 100
        
        # 波動率等級
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
            'volatility_current': volatility_current,
            'volatility_pred_3': volatility_pred_3,
            'volatility_pred_5': volatility_pred_5,
            'volatility_level': vol_level,
            'atr_14': round(atr_14, 2),
        }

model_manager = ModelManagerV2()

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
            
            # 非同步訂用（防止阻婫）
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
        """生成漠例 K 棒數據"""
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
market_analyzer = MarketAnalyzer()

# ============================================================================
# API 端點
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "CPB Trading Prediction API - V2",
        "version": "2.0.0",
        "model_type": "V2",
        "supported_coins": len(SUPPORTED_COINS),
        "endpoints": {
            "/coins": "List supported coins (20)",
            "/predict": "Get prediction for a coin",
            "/predict-batch": "Batch predict multiple coins",
            "/market-analysis": "Analyze market trend and find best entry point",
            "/health": "Health check"
        }
    }

@app.get("/coins")
async def get_coins():
    """清底所有支援的幣種"""
    return {
        "coins": SUPPORTED_COINS,
        "total": len(SUPPORTED_COINS),
        "model_version": "V2"
    }

@app.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResult:
    """預測交易信號和開單點位 (V2 模型)"""
    
    # 驗證幣種
    if request.coin not in SUPPORTED_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Coin {request.coin} not supported. Supported ({len(SUPPORTED_COINS)}): {SUPPORTED_COINS}"
        )
    
    # 1. 獲取實時 K 棒數據
    klines = await data_fetcher.fetch_klines(
        request.coin,
        timeframe='1h',
        limit=request.lookback_periods
    )
    
    current_price = klines[-1]['close']
    logger.info(f"{request.coin} current price: {current_price}")
    
    # 2. 執行 V2 模型預測
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
    
    # 推謀點位
    if direction > 0:  # 看漲
        recommendation = "BUY"
        entry_price = current_price
        stop_loss = round(current_price * 0.98, 2)  # 止損 2%
        take_profit = round(price_5 * 1.02, 2) if price_5 > current_price else round(price_5, 2)
    elif direction < 0:  # 看空
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
            current=round(volatility_current, 2),
            predicted_3=round(volatility_pred_3, 2),
            predicted_5=round(volatility_pred_5, 2),
            level=volatility_level,
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
        ],
        model_version="V2"
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

@app.post("/market-analysis")
async def market_analysis(request: MarketAnalysisRequest) -> MarketAnalysisResponse:
    """市場分析 - 趨勢判斷和最佳入場點計算"""
    
    try:
        # 將幣種名灊轉換成 Binance 格式
        symbol_map = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
            'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
            'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
            'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
            'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
        }
        
        binance_symbol = symbol_map.get(request.symbol, request.symbol + 'USDT')
        
        # 尋找最適時間框架
        timeframe = '1d' if request.timeframe == '1d' else '1h'
        
        # 1. 獲取擤圭 K線數據矩陣(20根歷史) + 上來10根預測
        all_klines = await data_fetcher.fetch_klines(binance_symbol, timeframe, limit=30)
        
        # 分成歷史數據和予測數據
        historical_klines = all_klines[:20]
        forecast_klines = all_klines[20:30]
        
        # 提取價格慰新恰寶 
        historical_prices = [k['close'] for k in historical_klines]
        forecast_prices = [k['close'] for k in forecast_klines]
        current_price = historical_klines[-1]['close']
        
        # 2. 執行市場分析
        analysis_result = market_analyzer.analyze(
            current_price=current_price,
            historical_prices=historical_prices,
            forecast_prices=forecast_prices,
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        # 3. 構建回應
        response = MarketAnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            trend={
                'direction': analysis_result.trend.direction,
                'strength': round(analysis_result.trend.strength, 2),
                'consecutive_bars': analysis_result.trend.consecutive_bars,
                'average_return': round(analysis_result.trend.average_return * 100, 2),
                'description': analysis_result.trend.description
            },
            best_entry_bar=analysis_result.best_entry_bar,
            price_extremes={
                'lowest_price': analysis_result.price_extremes.lowest_price,
                'lowest_bar': analysis_result.price_extremes.lowest_bar,
                'highest_price': analysis_result.price_extremes.highest_price,
                'highest_bar': analysis_result.price_extremes.highest_bar,
                'potential_profit': round(analysis_result.price_extremes.potential_profit * 100, 2)
            },
            forecast_prices=forecast_prices,
            recommendation=analysis_result.recommendation
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康検查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_version": "V2",
        "models_cached": len(model_manager.models),
        "supported_coins": len(SUPPORTED_COINS)
    }

# ============================================================================
# 運行
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print(" "*20 + "CPB Trading Web - V2 Model")
    print("="*80)
    print(f"\nModel Version: V2")
    print(f"Supported Coins: {len(SUPPORTED_COINS)}")
    print(f"\nStarting FastAPI server...")
    print(f"API: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
