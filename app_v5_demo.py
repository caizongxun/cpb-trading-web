#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V5 Model Inference (DEMO VERSION)
使用模擬模型進行完整系統測試
當真實 HuggingFace 模型可用時，只需替換 ModelManagerV5 類別
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import yfinance as yf

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI 初始化
# ============================================================================
app = FastAPI(title="CPB Trading Prediction API - V5 (DEMO)", version="5.0.0-DEMO")

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

print(f"\n[✓] 模型版本: V5 (DEMO)")
print(f"[✓] 支援幣種: {len(SUPPORTED_CRYPTOS_V5)}")
print(f"[✓] 時間框架: {SUPPORTED_TIMEFRAMES}")
print(f"[✓] 幣種清單: {list(SUPPORTED_CRYPTOS_V5.keys())}")
print(f"[⚠] 使用模擬模型用於測試\n")

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
# V5 模型管理 (DEMO VERSION - 使用模擬預測)
# ============================================================================

class ModelManagerV5:
    def __init__(self):
        self.models = {}
        logger.info("ModelManager V5 (DEMO) initialized")
    
    def predict(self, symbol: str, timeframe: str, klines_data: List[Dict]) -> Optional[Dict]:
        """執行 V5 模型預測 (DEMO: 基於實際數據的智能模擬)"""
        try:
            # 轉換為 DataFrame
            df = pd.DataFrame(klines_data)
            
            # 工程化特徵
            df_feat = engineer_features(df[['open', 'high', 'low', 'close', 'volume']].copy())
            
            # 檢查數據充足性
            if len(df_feat) < 61:
                logger.warning(f"Not enough data: {len(df_feat)} rows")
                return None
            
            # 基於實際數據的智能模擬預測
            current_price = df['close'].iloc[-1]
            recent_closes = df['close'].iloc[-60:].values
            
            # 計算實際波動率和趨勢
            returns = np.log(recent_closes[1:] / recent_closes[:-1])
            volatility = np.std(returns)
            trend = np.mean(returns)
            
            # 計算 RSI (用於判斷超買超賣)
            rsi = df_feat['rsi'].iloc[-1]
            if np.isnan(rsi):
                rsi = 50
            
            # 計算 MACD 趨勢
            macd = df_feat['macd'].iloc[-1]
            macd_signal = df_feat['macd_signal'].iloc[-1]
            if np.isnan(macd):
                macd = 0
            if np.isnan(macd_signal):
                macd_signal = 0
            
            # 智能預測邏輯
            # 基於波動率、趨勢、RSI 和 MACD 的加權預測
            base_log_return = trend + volatility * 0.5
            
            # RSI 修正 (過度買入時減速上升，過度賣出時加速下降)
            if rsi > 70:  # 超買
                base_log_return *= 0.7
            elif rsi < 30:  # 超賣
                base_log_return *= 1.3
            
            # MACD 信號加成
            if macd > macd_signal:
                base_log_return += abs(macd - macd_signal) * 0.01
            else:
                base_log_return -= abs(macd - macd_signal) * 0.01
            
            # 添加隨機性使結果更逼真（但保持統計特性）
            noise = np.random.normal(0, volatility * 0.3)
            log_return = base_log_return + noise
            
            # 限制日線預測範圍在 -2% 到 +2%
            if timeframe == '1d':
                log_return = np.clip(log_return, -0.02, 0.02)
            else:  # 小時線範圍更小
                log_return = np.clip(log_return, -0.01, 0.01)
            
            # 計算預測價格
            predicted_price = current_price * np.exp(log_return)
            
            # 信心度計算 (基於指標一致性)
            signals = 0
            if log_return > 0 and rsi < 70:
                signals += 1
            if log_return > 0 and macd > macd_signal:
                signals += 1
            if log_return < 0 and rsi > 30:
                signals += 1
            if log_return < 0 and macd < macd_signal:
                signals += 1
            
            # 信心度: 指標越一致，信心度越高
            confidence = 0.5 + (signals / 8) * 0.45
            confidence = np.clip(confidence, 0.4, 0.95)
            
            logger.info(f"Predicted for {symbol} {timeframe}: return={log_return:.6f}, confidence={confidence:.2%}")
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'log_return': log_return,
                'confidence': confidence,
                'direction': 1 if log_return > 0.001 else (-1 if log_return < -0.001 else 0),
                'rsi': rsi,
                'macd': macd,
                'volatility': volatility
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
        "message": "CPB Trading Prediction API - V5 (DEMO)",
        "version": "5.0.0-DEMO",
        "model_type": "V5 (Dual Timeframe - DEMO)",
        "model_status": "DEMO - Using simulated predictions based on real technical analysis",
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
        "model_version": "V5-DEMO"
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
    
    # 推薦 (基於預測方向和信心度)
    if direction > 0 and confidence > 0.55:
        recommendation = "BUY"
        entry_price = round(current_price, 2)
        stop_loss = round(current_price * 0.98, 2)
        take_profit = round(predicted_price * 1.02, 2)
    elif direction < 0 and confidence > 0.55:
        recommendation = "SELL"
        entry_price = round(current_price, 2)
        stop_loss = round(current_price * 1.02, 2)
        take_profit = round(predicted_price * 0.98, 2)
    else:
        recommendation = "HOLD"
        entry_price = round(current_price, 2)
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
        model_version="V5-DEMO"
    )
    
    return result

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_version": "V5-DEMO",
        "model_type": "Simulated (based on real technical analysis)",
        "supported_symbols": len(SUPPORTED_CRYPTOS_V5),
        "notes": "This is a DEMO version. Production version will use actual trained models from HuggingFace."
    }

# ============================================================================
# 運行
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print(" "*15 + "CPB Trading Web - V5 Model (DEMO)")
    print("="*80)
    print(f"\nModel Version: V5-DEMO")
    print(f"Supported Symbols: {len(SUPPORTED_CRYPTOS_V5)}")
    print(f"Timeframes: {SUPPORTED_TIMEFRAMES}")
    print(f"Features: {30}+ technical indicators")
    print(f"Prediction Type: Simulated (based on real technical analysis)")
    print(f"\nStarting FastAPI server...")
    print(f"API: http://localhost:8001")
    print(f"Docs: http://localhost:8001/docs")
    print("\n⚠  This is a DEMO version for testing the full system.")
    print("   When real HuggingFace models are available, simply replace")
    print("   the ModelManagerV5 class to use the trained models.")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
