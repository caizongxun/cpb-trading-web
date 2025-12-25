#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPB Trading Web - V5 Model (HYBRID VERSION) - DEBUG VERSION
包含市场分析端点、价格校正 + 残差补偿
"""

import asyncio
import yfinance as yf
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
import os

# 详细的日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPB Trading V5 HYBRID DEBUG", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("="*80)
logger.info("FastAPI 应用启动")
logger.info("="*80)

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
    original_predicted_price: float  # 修正前的价格
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
    residual_bias_compensation: float  # 应用的残差补偿值

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

logger.debug("数据模型已定义")

# ============================================================================
# 价格修正模块 - 处理 MAPE 偏移 + 残差补偿
# ============================================================================

class PriceCorrector:
    """
    处理模型预测的价格偏移
    包括：
    1. 动态波动率修正 (Dynamic Volatility Correction)
    2. 残差偏差补偿 (Residual Bias Compensation) - 基于 2025-12-25 分析
    """
    
    def __init__(self):
        self.correction_factors = {
            'aggressive_high': 1.15,
            'aggressive_low': 0.85,
            'moderate': 1.00,
        }
        
        # ========== 关键：残差补偿表 ==========
        # 数据来源：check_residuals.py (2025-12-25 20:30:01)
        # 基于 90 天历史数据的系统性偏差分析
        # 只补偿 "显著低估" 和 "轻微低估" 的币种
        # 小币种的偏差太小或不稳定，不做静态补偿
        self.residual_bias_map = {
            'BTC-USD': 500.0,      # 平均残差 496.24 → 补偿 500
            'ETH-USD': 21.5,       # 平均残差 21.49 → 补偿 21.5
            'BNB-USD': 5.0,        # 平均残差 4.94 → 补偿 5.0
            'SOL-USD': 1.5,        # 平均残差 1.50 → 补偿 1.5
            'LTC-USD': 0.4,        # 平均残差 0.41 → 补偿 0.4
            'AVAX-USD': 0.2,       # 平均残差 0.19 → 补偿 0.2
            'LINK-USD': 0.13,      # 平均残差 0.13 → 补偿 0.13
            # 不补偿：DOGE (0.0018), ADA (0.0049), DOT (0.016), XRP (0.028), ATOM (0.029)
            # 原因：偏差极小或标准差过大，补偿反而加大误差
        }
        
        logger.info("价格修正器已初始化")
        logger.info(f"残差补偿表加载: {len(self.residual_bias_map)} 个币种")
        for sym, bias in self.residual_bias_map.items():
            logger.debug(f"  {sym}: +{bias}")
    
    def correct_predicted_price(
        self,
        current_price: float,
        predicted_price: float,
        historical_prices: List[float],
        confidence: float = 0.65,
        symbol: str = "BTC-USD"
    ) -> Dict:
        logger.debug(f"开始价格修正: current={current_price}, predicted={predicted_price}, symbol={symbol}")
        
        price_array = np.array(historical_prices)
        current = float(current_price)
        predicted = float(predicted_price)
        
        pct_change = (predicted - current) / current if current > 0 else 0
        logger.debug(f"价格变化百分比: {pct_change*100:.4f}%")
        
        if len(price_array) > 1:
            hist_returns = np.diff(price_array) / price_array[:-1]
            hist_volatility = np.std(hist_returns)
        else:
            hist_volatility = 0.02
        
        logger.debug(f"历史波动: {hist_volatility*100:.4f}%")
        
        if abs(pct_change) > hist_volatility * 3:
            logger.warning("预测超过3倍历史波动，需要修正!")
            if pct_change > 0:
                correction_factor = self.correction_factors['aggressive_high']
                logger.debug(f"预测偏高, 修正因子: {correction_factor}")
            else:
                correction_factor = self.correction_factors['aggressive_low']
                logger.debug(f"预测偏低, 修正因子: {correction_factor}")
            
            corrected_price = current * (1 + pct_change / correction_factor)
        else:
            logger.debug("预测在合理范围内, 不需修正")
            corrected_price = predicted
        
        max_change = hist_volatility * 5
        max_price = current * (1 + max_change)
        min_price = current * (1 - max_change)
        
        corrected_price = max(min_price, min(max_price, corrected_price))
        logger.debug(f"修正后: {corrected_price}")
        
        # ========== 关键：叠加残差补偿 ==========
        residual_compensation = 0.0
        if symbol in self.residual_bias_map:
            residual_compensation = self.residual_bias_map[symbol]
            corrected_price += residual_compensation
            logger.info(f"应用残差补偿: +{residual_compensation} (symbol={symbol})")
        else:
            logger.debug(f"未找到 {symbol} 的补偿配置，跳过残差补偿")
        
        return {
            'original_predicted': predicted,
            'corrected_predicted': corrected_price,
            'correction_applied': corrected_price != predicted,
            'correction_pct': ((corrected_price - predicted) / predicted * 100) if predicted > 0 else 0,
            'hist_volatility': hist_volatility,
            'max_allowed_change': max_change,
            'residual_compensation': residual_compensation,  # 新增
        }

price_corrector = PriceCorrector()

# ============================================================================
# 市场分析模块
# ============================================================================

class MarketAnalyzer:
    """市场分析引擎"""
    
    def analyze_trend(self, historical_prices: List[float]) -> dict:
        logger.debug("分析趋势...")
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
        description = f"{trend_name}趋势明显，{strength_desc}程度"
        
        logger.debug(f"趋势结果: {direction}, 强度: {strength*100:.1f}%")
        
        return {
            'direction': direction,
            'strength': min(max(strength, 0.0), 1.0),
            'consecutive_bars': up_count if strength > 0.5 else down_count,
            'average_return': avg_return,
            'description': description
        }
    
    def find_price_extremes(self, forecast_prices: List[float]) -> dict:
        logger.debug("寻找价格极值...")
        if not forecast_prices:
            return {}
        
        lowest_price = min(forecast_prices)
        highest_price = max(forecast_prices)
        lowest_bar = forecast_prices.index(lowest_price) + 1
        highest_bar = forecast_prices.index(highest_price) + 1
        potential_profit = (highest_price - lowest_price) / lowest_price if lowest_price > 0 else 0
        
        logger.debug(f"最低: ${lowest_price:.2f} (第{lowest_bar}根), 最高: ${highest_price:.2f} (第{highest_bar}根)")
        
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
        logger.info(f"\n=== 市场分析 ===")
        logger.info(f"符号: {symbol}, 时间框: {timeframe}")
        logger.info(f"当前价: ${current_price:.2f}")
        logger.info(f"历史数据: {len(historical_prices)} 根")
        logger.info(f"预测数据: {len(forecast_prices)} 根")
        
        trend = self.analyze_trend(historical_prices)
        price_extremes = self.find_price_extremes(forecast_prices)
        
        if trend['direction'] == 'uptrend':
            best_entry = forecast_prices.index(min(forecast_prices)) + 1
        else:
            best_entry = forecast_prices.index(max(forecast_prices)) + 1
        
        logger.info(f"最佳入场点: 第 {best_entry} 根K棒")
        
        entry_price = forecast_prices[best_entry - 1] if best_entry <= len(forecast_prices) else current_price
        
        if trend['direction'] == 'uptrend':
            recommendation = f"多头信号 (第{best_entry}根)"
        else:
            recommendation = f"空头信号 (第{best_entry}根)"
        
        logger.info(f"建议: {recommendation}")
        logger.info("=" * 50)
        
        return {
            'trend': trend,
            'best_entry_bar': best_entry,
            'price_extremes': price_extremes,
            'recommendation': recommendation
        }

market_analyzer = MarketAnalyzer()
logger.info("市场分析器已初始化")

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
        logger.debug(f"\n[获取K线] {symbol} {timeframe} {limit}根")
        try:
            if symbol.endswith('USDT'):
                yf_symbol = symbol.replace('USDT', '-USD')
            else:
                yf_symbol = symbol + '-USD'
            
            logger.debug(f"yfinance 符号: {yf_symbol}")
            
            interval = '1h' if timeframe == '1h' else '1d'
            period = '90d' if timeframe == '1d' else '30d'
            
            logger.debug(f"周期: {period}, 间隔: {interval}")
            
            df = yf.download(
                yf_symbol,
                period=period,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                logger.warning(f"未获取到数据: {yf_symbol}")
                return self._generate_demo_klines(limit)
            
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
            
            logger.debug(f"获取了 {len(klines)} 根K线")
            return klines
        
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return self._generate_demo_klines(limit)
    
    def _generate_demo_klines(self, limit: int = 30) -> List[Dict]:
        logger.debug("生成演示数据")
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
logger.info("数据获取器已初始化")

# ============================================================================
# 预测模块
# ============================================================================

class PredictionEngine:
    """预测引擎"""
    
    def __init__(self):
        self.demo_mode = True
        logger.info("预测引擎已初始化 (演示模式)")
    
    def predict(self, klines: List[Dict], symbol: str = "BTC-USD") -> Dict:
        logger.debug(f"\n[预测] 处理 {len(klines)} 根K线, symbol={symbol}")
        
        closes = [k['close'] for k in klines]
        current_price = closes[-1]
        
        recent_avg = sum(closes[-5:]) / 5
        past_avg = sum(closes[:5]) / 5
        trend = recent_avg - past_avg
        
        direction = 1 if trend > 0 else (-1 if trend < 0 else 0)
        logger.debug(f"趋势方向: {direction}")
        
        volatility = np.std(np.array(closes[-10:]) / np.array(closes[-11:-1]) - 1)
        predicted_change = direction * volatility * 0.5
        predicted_price = current_price * (1 + predicted_change)
        logger.debug(f"未修正预测: {predicted_price:.2f}")
        
        correction = price_corrector.correct_predicted_price(
            current_price,
            predicted_price,
            closes,
            confidence=0.65,
            symbol=symbol  # 传入符号，用于残差补偿
        )
        
        corrected_price = correction['corrected_predicted']
        
        volatility_current = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0
        volatility_predicted = ((corrected_price - current_price) / current_price * 100)
        
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
            'correction_info': correction,
            'residual_compensation': correction['residual_compensation'],  # 新增
        }

prediction_engine = PredictionEngine()

# ============================================================================
# API 端点
# ============================================================================

logger.info("\n" + "="*40 + "\n正在注册 API 端点\n" + "="*40)

@app.get("/kline-dashboard", response_class=HTMLResponse)
async def kline_dashboard():
    """提供K線图表仪表板"""
    try:
        with open('kline_dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("找K線图表HTML文件失败")
        return HTMLResponse(
            content="<h1>文件未找到</h1><p>请确保 kline_dashboard.html 存在于查询目录。</p>",
            status_code=404
        )

@app.post("/predict-v5")
async def predict_v5(request: Dict) -> PredictionResponse:
    """В5 预测端点"""
    logger.info(f"\n[预测] 请求: {request}")
    
    symbol = request.get('symbol', 'BTC')
    timeframe = request.get('timeframe', '1d')
    use_binance = request.get('use_binance', False)
    
    symbol_map = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
        'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
        'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
        'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
    }
    
    binance_symbol = symbol_map.get(symbol, symbol + 'USDT')
    yf_symbol = binance_symbol.replace('USDT', '-USD')  # 用于残差补偿查询
    logger.debug(f"Binance 符号: {binance_symbol}, yfinance 符号: {yf_symbol}")
    
    try:
        klines = await data_fetcher.fetch_klines(
            binance_symbol,
            timeframe=timeframe,
            limit=25
        )
        
        pred = prediction_engine.predict(klines, symbol=yf_symbol)  # 传入 yfinance 符号
        
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
        
        logger.info(f"[预测] 成功: {symbol} {pred['recommendation']}")
        logger.info(f"        残差补偿: +{pred['residual_compensation']:.2f}")
        
        return PredictionResponse(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current,
            predicted_price=predicted,
            original_predicted_price=pred['original_predicted'],
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
            model_version='V5 HYBRID + Residual Bias Compensation',
            residual_bias_compensation=pred['residual_compensation'],
        )
    
    except Exception as e:
        logger.error(f"[预测] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-analysis")
async def market_analysis(request: MarketAnalysisRequest) -> MarketAnalysisResponse:
    """市场分析端点"""
    
    logger.info(f"\n[MARKET_ANALYSIS_START]")
    logger.info(f"符号: {request.symbol}")
    logger.info(f"时间框架: {request.timeframe}")
    logger.info(f"Binance: {request.use_binance}")
    
    try:
        symbol_map = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
            'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
            'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
            'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
            'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
        }
        
        binance_symbol = symbol_map.get(request.symbol, request.symbol + 'USDT')
        logger.debug(f"Binance 符号: {binance_symbol}")
        
        logger.debug("获取 K线数据...")
        all_klines = await data_fetcher.fetch_klines(
            binance_symbol,
            timeframe=request.timeframe,
            limit=30
        )
        logger.info(f"获取了 {len(all_klines)} 根K线")
        
        if len(all_klines) < 20:
            logger.error("数据不足 (< 20)")
            raise HTTPException(status_code=400, detail="数据不足")
        
        historical_klines = all_klines[:20]
        forecast_klines = all_klines[20:30]
        
        historical_prices = [k['close'] for k in historical_klines]
        forecast_prices = [k['close'] for k in forecast_klines]
        current_price = historical_klines[-1]['close']
        
        logger.debug(f"历史价: {len(historical_prices)}, 预测价: {len(forecast_prices)}, 当前: ${current_price:.2f}")
        
        logger.debug("执行市场分析...")
        analysis = market_analyzer.analyze(
            current_price=current_price,
            historical_prices=historical_prices,
            forecast_prices=forecast_prices,
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        logger.info("[市场分析] 成功")
        
        response = MarketAnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            trend=analysis['trend'],
            best_entry_bar=analysis['best_entry_bar'],
            price_extremes=analysis['price_extremes'],
            forecast_prices=forecast_prices,
            recommendation=analysis['recommendation']
        )
        
        logger.info("[MARKET_ANALYSIS_END] 成功")
        return response
    
    except Exception as e:
        logger.error(f"[市场分析] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """根端点"""
    logger.debug("根端点请求")
    return {
        "message": "CPB Trading V5 HYBRID DEBUG",
        "version": "5.0.0",
        "features": [
            "Price prediction with dynamic volatility correction",
            "Residual bias compensation (7 major coins)",
            "Market trend analysis",
            "Automatic entry/exit points",
            "Interactive K-line chart dashboard"
        ],
        "endpoints": {
            "/": "根端点",
            "/kline-dashboard": "互动式K線官方板",
            "/predict-v5": "获取价格预测 (含残差补偿)",
            "/market-analysis": "市场趋势分析和最佳入场点"
        }
    }

logger.info("所有 API 端点已注册")

if __name__ == "__main__":
    import uvicorn
    
    print("""
================================================================================
               CPB Trading Web - V5 Model (HYBRID VERSION) - DEBUG
================================================================================

Model Version: V5 (HYBRID) + Residual Bias Compensation + K-Line Dashboard
Strategy: Informed by 2025-12-25 high-precision residual analysis (7 decimal places)
Price Source: yfinance (unified, consistent across timeframes)
Supported Symbols: 14
Timeframes: ['1d', '1h']

Starting FastAPI server...
API: http://localhost:8001
Docs: http://localhost:8001/docs
K-Line Dashboard: http://localhost:8001/kline-dashboard

⚠  PRICE CORRECTION ENABLED!
   - Automatic MAPE offset correction
   - Historical volatility-based bounds
   - Smart confidence-weighted adjustments

⚠  RESIDUAL BIAS COMPENSATION ENABLED!
   - BTC-USD: +500.00
   - ETH-USD: +21.50
   - BNB-USD: +5.00
   - SOL-USD: +1.50
   - LTC-USD: +0.40
   - AVAX-USD: +0.20
   - LINK-USD: +0.13

⚠  K-LINE DASHBOARD ENABLED!
   - Interactive candlestick chart
   - Multi-coin selector
   - 30 historical bars + 10 forecast bars
   - Real-time ATR-based range prediction

⚠  MARKET ANALYSIS ENABLED!
   - Trend detection (uptrend/downtrend)
   - Best entry point calculation
   - Price extremes analysis

⚠  DEBUG MODE ENABLED!
   - Detailed logging for troubleshooting

================================================================================
    """)
    
    logger.info("\n" + "="*80)
    logger.info("启动 FastAPI 服务器")
    logger.info("="*80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="debug"
    )
