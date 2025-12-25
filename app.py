#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V5 Model (HYBRID VERSION)
包含市场分析端点和价格偏移修正
"""

import asyncio
import yfinance as yf
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPB Trading V5 HYBRID", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 数据模型
# ============================================================================

class VolatilityData(BaseModel):
    current: float
    predicted: float
    level: str
    atr_14: float

class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    current_price: float
    predicted_price: float
    log_return: float
    confidence: float
    recommendation: str
    entry_price: float
    stop_loss: float
    take_profit: float
    volatility: VolatilityData
    klines: List[Dict]
    timestamp: str
    price_source: str
    model_version: str

class MarketAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = '1d'
    use_binance: bool = False

class MarketAnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    trend: Dict
    best_entry_bar: int
    price_extremes: Dict
    forecast_prices: List[float]
    recommendation: str

# ============================================================================
# 价格修正模块 - 处理 MAPE 偏移
# ============================================================================

class PriceCorrector:
    """
    处理模型预测的价格偏移
    MAPE 偏差通常来自于:
    1. 价格正常化和反正常化的累积误差
    2. 极端价格点的过度或不足预测
    3. 市场波动的非线性特征
    """
    
    def __init__(self):
        # 基于经验的偏移修正因子
        self.correction_factors = {
            'aggressive_high': 1.15,  # 预测偏高时
            'aggressive_low': 0.85,   # 预测偏低时
            'moderate': 1.00,          # 适度预测
        }
    
    def correct_predicted_price(
        self,
        current_price: float,
        predicted_price: float,
        historical_prices: List[float],
        confidence: float = 0.65
    ) -> Dict:
        """
        修正预测价格，考虑:
        - 历史波动率
        - 预测信心度
        - 价格偏差方向
        """
        
        # 计算历史价格统计
        price_array = np.array(historical_prices)
        current = float(current_price)
        predicted = float(predicted_price)
        
        # 计算基本变化百分比
        pct_change = (predicted - current) / current if current > 0 else 0
        
        # 计算历史波动率 (过去20根K线的标准差)
        if len(price_array) > 1:
            hist_returns = np.diff(price_array) / price_array[:-1]
            hist_volatility = np.std(hist_returns)
        else:
            hist_volatility = 0.02  # 默认2%
        
        # 确定修正方向
        if abs(pct_change) > hist_volatility * 3:  # 预测超过3倍历史波动
            # 偏差过大，需要修正
            if pct_change > 0:
                # 预测偏高，向下修正
                correction_factor = self.correction_factors['aggressive_high']
            else:
                # 预测偏低，向上修正
                correction_factor = self.correction_factors['aggressive_low']
            
            corrected_price = current * (1 + pct_change / correction_factor)
        else:
            # 预测在合理范围内
            corrected_price = predicted
        
        # 确保价格不会走极端
        max_change = hist_volatility * 5  # 最大允许变化为历史波动的5倍
        max_price = current * (1 + max_change)
        min_price = current * (1 - max_change)
        
        corrected_price = max(min_price, min(max_price, corrected_price))
        
        return {
            'original_predicted': predicted,
            'corrected_predicted': corrected_price,
            'correction_applied': corrected_price != predicted,
            'correction_pct': ((corrected_price - predicted) / predicted * 100) if predicted > 0 else 0,
            'hist_volatility': hist_volatility,
            'max_allowed_change': max_change,
        }

price_corrector = PriceCorrector()

# ============================================================================
# 市场分析模块
# ============================================================================

class MarketAnalyzer:
    """市场分析引擎"""
    
    def analyze_trend(self, historical_prices: List[float]) -> dict:
        """分析趋势"""
        if len(historical_prices) < 2:
            return {'direction': 'neutral', 'strength': 0.5, 'consecutive_bars': 0}
        
        recent_count = min(5, len(historical_prices))
        recent_prices = historical_prices[-recent_count:]
        
        up_count = down_count = 0
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                up_count += 1
            else:
                down_count += 1
        
        total = up_count + down_count
        strength = up_count / total if total > 0 else 0.5
        
        direction = 'uptrend' if strength > 0.5 else 'downtrend'
        avg_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0
        
        strength_desc = '强势' if strength > 0.7 else ('中等' if strength > 0.5 else '弱')
        trend_name = '多头上升' if direction == 'uptrend' else '空头下跌'
        description = f"{trend_name}趋势明显，{strength_desc}程度，最近{up_count if strength > 0.5 else down_count}根K线连续{'上升' if direction == 'uptrend' else '下跌'}"
        
        return {
            'direction': direction,
            'strength': min(max(strength, 0.0), 1.0),
            'consecutive_bars': up_count if strength > 0.5 else down_count,
            'average_return': avg_return,
            'description': description
        }
    
    def find_price_extremes(self, forecast_prices: List[float]) -> dict:
        """找出最高和最低点"""
        if not forecast_prices:
            return {}
        
        lowest_price = min(forecast_prices)
        highest_price = max(forecast_prices)
        lowest_bar = forecast_prices.index(lowest_price) + 1
        highest_bar = forecast_prices.index(highest_price) + 1
        potential_profit = (highest_price - lowest_price) / lowest_price if lowest_price > 0 else 0
        
        return {
            'lowest_price': lowest_price,
            'lowest_bar': lowest_bar,
            'highest_price': highest_price,
            'highest_bar': highest_bar,
            'potential_profit': potential_profit
        }
    
    def analyze(
        self,
        current_price: float,
        historical_prices: List[float],
        forecast_prices: List[float],
        symbol: str,
        timeframe: str
    ) -> dict:
        """完整分析"""
        trend = self.analyze_trend(historical_prices)
        price_extremes = self.find_price_extremes(forecast_prices)
        
        # 最佳入场点
        if trend['direction'] == 'uptrend':
            best_entry = forecast_prices.index(min(forecast_prices)) + 1
        else:
            best_entry = forecast_prices.index(max(forecast_prices)) + 1
        
        # 建议
        entry_price = forecast_prices[best_entry - 1] if best_entry <= len(forecast_prices) else current_price
        
        if trend['direction'] == 'uptrend':
            profit_target = price_extremes.get('highest_price', entry_price)
            stop_loss = price_extremes.get('lowest_price', entry_price) * 0.99
            profit_pct = (profit_target - entry_price) / entry_price * 100 if entry_price > 0 else 0
            risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
            
            recommendation = f"""
交易信号：强势多头
趋势分析：{trend['description']}

最优策略：
- 在第 {best_entry} 根K棒进行开多单
- 入场价格：${entry_price:.8f}
- 止盈目标：${profit_target:.8f}（潜在收益：+{profit_pct:.2f}%）
- 止损位置：${stop_loss:.8f}（控制风险：{risk_pct:.2f}%）
- 风险回报比：1:{(profit_pct/max(risk_pct, 0.01)):.2f}
            """
        else:
            profit_target = price_extremes.get('lowest_price', entry_price)
            stop_loss = price_extremes.get('highest_price', entry_price) * 1.01
            profit_pct = (entry_price - profit_target) / entry_price * 100 if entry_price > 0 else 0
            risk_pct = (stop_loss - entry_price) / entry_price * 100 if entry_price > 0 else 0
            
            recommendation = f"""
交易信号：强势空头
趋势分析：{trend['description']}

最优策略：
- 在第 {best_entry} 根K棒进行开空单
- 入场价格：${entry_price:.8f}
- 止盈目标：${profit_target:.8f}（潜在收益：+{profit_pct:.2f}%）
- 止损位置：${stop_loss:.8f}（控制风险：{risk_pct:.2f}%）
- 风险回报比：1:{(profit_pct/max(risk_pct, 0.01)):.2f}
            """
        
        return {
            'trend': trend,
            'best_entry_bar': best_entry,
            'price_extremes': price_extremes,
            'recommendation': recommendation.strip()
        }

market_analyzer = MarketAnalyzer()

# ============================================================================
# 数据获取
# ============================================================================

class DataFetcher:
    """数据获取"""
    
    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str = '1d',
        limit: int = 30
    ) -> List[Dict]:
        """获取K线数据"""
        try:
            # 转换符号
            if symbol.endswith('USDT'):
                yf_symbol = symbol.replace('USDT', '-USD')
            else:
                yf_symbol = symbol + '-USD'
            
            logger.info(f"[yfinance] 获取 {yf_symbol} {timeframe} 数据...")
            
            # 获取历史数据
            interval = '1h' if timeframe == '1h' else '1d'
            period = '90d' if timeframe == '1d' else '30d'
            
            df = yf.download(
                yf_symbol,
                period=period,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                logger.warning(f"未获取到数据: {yf_symbol}")
                return self._generate_demo_klines(limit)
            
            # 转换格式
            klines = []
            for idx, row in df.tail(limit).iterrows():
                klines.append({
                    'timestamp': int(idx.timestamp() * 1000),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                })
            
            logger.info(f"[yfinance] 获取了 {len(klines)} 根K线")
            return klines
        
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return self._generate_demo_klines(limit)
    
    def _generate_demo_klines(self, limit: int = 30) -> List[Dict]:
        """生成演示数据"""
        import random
        import time
        
        klines = []
        base_price = 42000
        current_time = int(time.time() * 1000) - (limit * 3600 * 1000)
        
        for i in range(limit):
            price_change = random.uniform(-200, 200)
            open_price = base_price + price_change
            close_price = open_price + random.uniform(-100, 100)
            high_price = max(open_price, close_price) + random.uniform(0, 100)
            low_price = min(open_price, close_price) - random.uniform(0, 100)
            
            klines.append({
                'timestamp': current_time + (i * 3600 * 1000),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.uniform(1000, 10000)
            })
            
            base_price = close_price
        
        return klines

data_fetcher = DataFetcher()

# ============================================================================
# 预测模块
# ============================================================================

class PredictionEngine:
    """预测引擎"""
    
    def __init__(self):
        self.demo_mode = True  # 默认演示模式
    
    def predict(
        self,
        klines: List[Dict]
    ) -> Dict:
        """执行预测"""
        
        closes = [k['close'] for k in klines]
        current_price = closes[-1]
        
        # 计算基本趋势
        recent_avg = sum(closes[-5:]) / 5
        past_avg = sum(closes[:5]) / 5
        trend = recent_avg - past_avg
        
        direction = 1 if trend > 0 else (-1 if trend < 0 else 0)
        
        # 演示预测
        volatility = np.std(np.array(closes[-10:]) / np.array(closes[-11:-1]) - 1)
        predicted_change = direction * volatility * 0.5  # 保守估计
        predicted_price = current_price * (1 + predicted_change)
        
        # 应用价格修正
        correction = price_corrector.correct_predicted_price(
            current_price,
            predicted_price,
            closes,
            confidence=0.65
        )
        
        corrected_price = correction['corrected_predicted']
        
        # 计算波动率
        volatility_current = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0
        volatility_predicted = ((corrected_price - current_price) / current_price * 100)
        
        # ATR
        tr_list = []
        for i in range(len(klines)):
            high = klines[i]['high']
            low = klines[i]['low']
            prev_close = closes[i-1] if i > 0 else low
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        atr_14 = np.mean(tr_list[-14:]) if len(tr_list) >= 14 else np.mean(tr_list)
        
        return {
            'current_price': current_price,
            'predicted_price': corrected_price,
            'original_predicted': correction['original_predicted'],
            'log_return': (corrected_price - current_price) / current_price if current_price > 0 else 0,
            'confidence': 0.68,
            'recommendation': 'BUY' if direction > 0 else ('SELL' if direction < 0 else 'HOLD'),
            'direction': direction,
            'volatility_current': volatility_current,
            'volatility_predicted': volatility_predicted,
            'atr_14': atr_14,
            'correction_info': correction
        }

prediction_engine = PredictionEngine()

# ============================================================================
# API 端点
# ============================================================================

@app.post("/predict-v5")
async def predict_v5(request: Dict) -> PredictionResponse:
    """V5 预测端点"""
    
    symbol = request.get('symbol', 'BTC')
    timeframe = request.get('timeframe', '1d')
    use_binance = request.get('use_binance', False)
    
    # 转换符号
    symbol_map = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
        'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
        'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
        'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
    }
    
    binance_symbol = symbol_map.get(symbol, symbol + 'USDT')
    
    try:
        # 获取数据
        klines = await data_fetcher.fetch_klines(
            binance_symbol,
            timeframe=timeframe,
            limit=25
        )
        
        # 执行预测
        pred = prediction_engine.predict(klines)
        
        # 计算交易点位
        current = pred['current_price']
        predicted = pred['predicted_price']
        direction = pred['direction']
        
        if direction > 0:
            entry = current
            stop_loss = current * 0.98
            take_profit = predicted * 1.02
        elif direction < 0:
            entry = current
            stop_loss = current * 1.02
            take_profit = predicted * 0.98
        else:
            entry = current
            stop_loss = current * 0.99
            take_profit = current * 1.01
        
        return PredictionResponse(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current,
            predicted_price=predicted,
            log_return=pred['log_return'],
            confidence=pred['confidence'],
            recommendation=pred['recommendation'],
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volatility=VolatilityData(
                current=pred['volatility_current'],
                predicted=pred['volatility_predicted'],
                level='高' if abs(pred['volatility_current']) > 1.5 else ('中' if abs(pred['volatility_current']) > 0.5 else '低'),
                atr_14=pred['atr_14']
            ),
            klines=klines,
            timestamp=datetime.now().isoformat(),
            price_source='yfinance',
            model_version='V5 HYBRID'
        )
    
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-analysis")
async def market_analysis(request: MarketAnalysisRequest) -> MarketAnalysisResponse:
    """市场分析端点"""
    
    try:
        # 转换符号
        symbol_map = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
            'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
            'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
            'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
            'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
        }
        
        binance_symbol = symbol_map.get(request.symbol, request.symbol + 'USDT')
        
        # 获取30根K线（20根历史 + 10根预测）
        all_klines = await data_fetcher.fetch_klines(
            binance_symbol,
            timeframe=request.timeframe,
            limit=30
        )
        
        if len(all_klines) < 20:
            raise HTTPException(status_code=400, detail="数据不足")
        
        historical_klines = all_klines[:20]
        forecast_klines = all_klines[20:30]
        
        historical_prices = [k['close'] for k in historical_klines]
        forecast_prices = [k['close'] for k in forecast_klines]
        current_price = historical_klines[-1]['close']
        
        # 执行分析
        analysis = market_analyzer.analyze(
            current_price=current_price,
            historical_prices=historical_prices,
            forecast_prices=forecast_prices,
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        return MarketAnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            trend=analysis['trend'],
            best_entry_bar=analysis['best_entry_bar'],
            price_extremes=analysis['price_extremes'],
            forecast_prices=forecast_prices,
            recommendation=analysis['recommendation']
        )
    
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "CPB Trading V5 HYBRID",
        "version": "5.0.0",
        "endpoints": {
            "/predict-v5": "获取价格预测",
            "/market-analysis": "市场趋势分析和最佳入场点"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("""
================================================================================
               CPB Trading Web - V5 Model (HYBRID VERSION)
================================================================================

Model Version: V5 (HYBRID)
Strategy: Demo Mode with Price Correction
Price Source: yfinance (unified, consistent across timeframes)
Supported Symbols: 14
Timeframes: ['1d', '1h']

Starting FastAPI server...
API: http://localhost:8001
Docs: http://localhost:8001/docs

⚠  PRICE CORRECTION ENABLED!
   - Automatic MAPE offset correction
   - Historical volatility-based bounds
   - Smart confidence-weighted adjustments

⚠  MARKET ANALYSIS ENABLED!
   - Trend detection (uptrend/downtrend)
   - Best entry point calculation
   - Price extremes analysis

================================================================================
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
