#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市場分析 API 路由
相容 V5 HYBRID 版本直接整合
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# 數據模型
# ============================================================================

class MarketAnalysisRequest(BaseModel):
    symbol: str  # 幣種 (e.g., 'BTC', 'ETH')
    timeframe: str = '1d'  # 時間框架 ('1h', '1d')
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
# 市場分析引擎
# ============================================================================

class TrendData:
    def __init__(self, direction, strength, consecutive_bars, average_return, description):
        self.direction = direction
        self.strength = strength
        self.consecutive_bars = consecutive_bars
        self.average_return = average_return
        self.description = description

class PriceExtremesData:
    def __init__(self, lowest_price, lowest_bar, highest_price, highest_bar, potential_profit):
        self.lowest_price = lowest_price
        self.lowest_bar = lowest_bar
        self.highest_price = highest_price
        self.highest_bar = highest_bar
        self.potential_profit = potential_profit

class MarketAnalyzer:
    """市場分析引擎"""
    
    def __init__(self):
        self.min_trend_strength = 0.3
        self.min_consecutive = 3
    
    def analyze_trend(self, historical_prices: List[float], forecast_prices: List[float]) -> TrendData:
        """分析趨勢方向和強度"""
        recent_count = min(5, len(historical_prices))
        recent_prices = historical_prices[-recent_count:]
        
        up_count = 0
        down_count = 0
        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                up_count += 1
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            else:
                down_count += 1
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
        
        total = up_count + down_count
        strength = up_count / total if total > 0 else 0.5
        
        if strength > 0.5:
            direction = 'uptrend'
            consecutive = max_consecutive_up
            avg_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0
        else:
            direction = 'downtrend'
            consecutive = max_consecutive_down
            avg_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0
        
        if strength > 0.7:
            strength_desc = '強勢'
        elif strength > 0.5:
            strength_desc = '中等'
        else:
            strength_desc = '弱'
        
        trend_name = '多頭上升' if direction == 'uptrend' else '空頭下跌'
        description = f"{trend_name}趨勢明顯，{strength_desc}程度，最近{consecutive}根K線連續{('上升' if direction == 'uptrend' else '下跌')}"
        
        return TrendData(
            direction=direction,
            strength=max(min(strength, 1.0), 0.0),
            consecutive_bars=consecutive,
            average_return=avg_return,
            description=description
        )
    
    def find_price_extremes(self, forecast_prices: List[float]) -> PriceExtremesData:
        """找出未來10根K棒中的最高和最低價格"""
        lowest_price = min(forecast_prices)
        highest_price = max(forecast_prices)
        
        lowest_bar = forecast_prices.index(lowest_price) + 1
        highest_bar = forecast_prices.index(highest_price) + 1
        
        potential_profit = (highest_price - lowest_price) / lowest_price
        
        return PriceExtremesData(
            lowest_price=lowest_price,
            lowest_bar=lowest_bar,
            highest_price=highest_price,
            highest_bar=highest_bar,
            potential_profit=potential_profit
        )
    
    def calculate_best_entry(self, trend: TrendData, forecast_prices: List[float]) -> int:
        """計算最佳入場點位"""
        if trend.direction == 'uptrend':
            best_bar = forecast_prices.index(min(forecast_prices)) + 1
        else:
            best_bar = forecast_prices.index(max(forecast_prices)) + 1
        
        return best_bar
    
    def generate_recommendation(self, trend: TrendData, best_entry_bar: int, 
                               price_extremes: PriceExtremesData, forecast_prices: List[float]) -> str:
        """生成詳細的交易建議"""
        entry_price = forecast_prices[best_entry_bar - 1]
        
        if trend.direction == 'uptrend':
            profit_target = price_extremes.highest_price
            stop_loss = price_extremes.lowest_price * 0.99
            profit_pct = (profit_target - entry_price) / entry_price * 100
            risk_pct = (entry_price - stop_loss) / entry_price * 100
            
            recommendation = f"""
交易信號：強勢多頭
趨勢分析：{trend.description}

最優策略：
- 在第 {best_entry_bar} 根K棒進行開多單
- 入場價格：${entry_price:.8f}
- 止盈目標：${profit_target:.8f}（潛在收益：+{profit_pct:.2f}%）
- 止損位置：${stop_loss:.8f}（控制風險：{risk_pct:.2f}%）
- 風險回報比：1:{(profit_pct/risk_pct):.2f}
            """
        else:
            profit_target = price_extremes.lowest_price
            stop_loss = price_extremes.highest_price * 1.01
            profit_pct = (entry_price - profit_target) / entry_price * 100
            risk_pct = (stop_loss - entry_price) / entry_price * 100
            
            recommendation = f"""
交易信號：強勢空頭
趨勢分析：{trend.description}

最優策略：
- 在第 {best_entry_bar} 根K棒進行開空單
- 入場價格：${entry_price:.8f}
- 止盈目標：${profit_target:.8f}（潛在收益：+{profit_pct:.2f}%）
- 止損位置：${stop_loss:.8f}（控制風險：{risk_pct:.2f}%）
- 風險回報比：1:{(profit_pct/risk_pct):.2f}
            """
        
        return recommendation.strip()
    
    def analyze(self, current_price: float, historical_prices: List[float],
               forecast_prices: List[float], symbol: str, timeframe: str) -> Dict:
        """執行完整的市場分析"""
        trend = self.analyze_trend(historical_prices, forecast_prices)
        price_extremes = self.find_price_extremes(forecast_prices)
        best_entry_bar = self.calculate_best_entry(trend, forecast_prices)
        recommendation = self.generate_recommendation(trend, best_entry_bar, price_extremes, forecast_prices)
        
        return {
            'trend': {
                'direction': trend.direction,
                'strength': round(trend.strength, 2),
                'consecutive_bars': trend.consecutive_bars,
                'average_return': round(trend.average_return * 100, 2),
                'description': trend.description
            },
            'best_entry_bar': best_entry_bar,
            'price_extremes': {
                'lowest_price': round(price_extremes.lowest_price, 8),
                'lowest_bar': price_extremes.lowest_bar,
                'highest_price': round(price_extremes.highest_price, 8),
                'highest_bar': price_extremes.highest_bar,
                'potential_profit': round(price_extremes.potential_profit * 100, 2)
            },
            'recommendation': recommendation
        }

# ============================================================================
# API 路由
# ============================================================================

router = APIRouter(prefix="", tags=["market-analysis"])
market_analyzer = MarketAnalyzer()

# 你需要供應一個 data_fetcher 實例
def setup_market_analysis_routes(app, data_fetcher):
    """
    需要從你的 V5 app.py 中訪問，
    传入 FastAPI app 和 data_fetcher 實例
    
    使用步驥：
    from market_analysis_api import setup_market_analysis_routes
    setup_market_analysis_routes(app, data_fetcher)
    """
    
    @app.post("/market-analysis")
    async def market_analysis(request: MarketAnalysisRequest) -> MarketAnalysisResponse:
        """市場分析 - 趨勢判斷和最佳入場點計算"""
        
        try:
            symbol_map = {
                'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
                'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
                'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
                'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
                'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
            }
            
            binance_symbol = symbol_map.get(request.symbol, request.symbol + 'USDT')
            timeframe = '1d' if request.timeframe == '1d' else '1h'
            
            logger.info(f"Market analysis request: {request.symbol} ({timeframe})")
            
            # 1. 獲取數據 (30根K棒: 20佥歷史 + 10予測)
            all_klines = await data_fetcher.fetch_klines(
                binance_symbol,
                timeframe=timeframe,
                limit=30
            )
            
            if len(all_klines) < 20:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data: got {len(all_klines)} klines, need at least 20"
                )
            
            historical_klines = all_klines[:20]
            forecast_klines = all_klines[20:30]
            
            historical_prices = [k['close'] for k in historical_klines]
            forecast_prices = [k['close'] for k in forecast_klines]
            current_price = historical_klines[-1]['close']
            
            # 2. 執行市場分析
            analysis = market_analyzer.analyze(
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
                trend=analysis['trend'],
                best_entry_bar=analysis['best_entry_bar'],
                price_extremes=analysis['price_extremes'],
                forecast_prices=forecast_prices,
                recommendation=analysis['recommendation']
            )
            
            logger.info(f"Market analysis completed: {request.symbol}")
            return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Market analysis error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
