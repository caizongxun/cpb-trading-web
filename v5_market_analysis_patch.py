#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5 Market Analysis Patch

此文件是一個准推載的 patch，請直接複製下方所有代碼到你的 V5 app.py 中。

步驥:
1. 從本文件複製代碼
2. 貼到你的 app.py (在 if __name__ == '__main__' 前)
3. 重新啟動 API
"""

# ============================================================================
# 粗推到 app.py: 在你的所有 import 之後添加
# ============================================================================

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 粗推到 app.py: 數據模型段
# ============================================================================

class MarketAnalysisRequest:
    def __init__(self, symbol: str, timeframe: str = '1d', use_binance: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_binance = use_binance

# ============================================================================
# 粗推到 app.py: 市場分析引擎
# ============================================================================

class MarketAnalyzer:
    """市場分析引擎"""
    
    def __init__(self):
        logger.info("[Market Analyzer] Initialized")
    
    def analyze_trend(self, historical_prices: List[float]) -> dict:
        """分析趨勢方向和強度"""
        if len(historical_prices) < 2:
            return {'direction': 'neutral', 'strength': 0.5, 'consecutive_bars': 0}
        
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
            elif recent_prices[i] < recent_prices[i-1]:
                down_count += 1
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
        
        total = up_count + down_count
        strength = up_count / total if total > 0 else 0.5
        
        direction = 'uptrend' if strength > 0.5 else 'downtrend'
        consecutive = max_consecutive_up if strength > 0.5 else max_consecutive_down
        avg_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0
        
        if strength > 0.7:
            strength_desc = '強勢'
        elif strength > 0.5:
            strength_desc = '中等'
        else:
            strength_desc = '弱'
        
        trend_name = '多頭上升' if direction == 'uptrend' else '空頭下跌'
        description = f"{trend_name}趨勢明顯，{strength_desc}程度，最近{consecutive}根K線連續{('上升' if direction == 'uptrend' else '下跌')}"
        
        return {
            'direction': direction,
            'strength': min(max(strength, 0.0), 1.0),
            'consecutive_bars': consecutive,
            'average_return': avg_return,
            'description': description
        }
    
    def find_price_extremes(self, forecast_prices: List[float]) -> dict:
        """找出未來10根K棒中的最高和最低價格"""
        if not forecast_prices:
            return {}
        
        lowest_price = min(forecast_prices)
        highest_price = max(forecast_prices)
        lowest_bar = forecast_prices.index(lowest_price) + 1
        highest_bar = forecast_prices.index(highest_price) + 1
        potential_profit = (highest_price - lowest_price) / lowest_price if lowest_price > 0 else 0
        
        return {
            'lowest_price': round(lowest_price, 8),
            'lowest_bar': lowest_bar,
            'highest_price': round(highest_price, 8),
            'highest_bar': highest_bar,
            'potential_profit': round(potential_profit * 100, 2)
        }
    
    def calculate_best_entry(self, trend: dict, forecast_prices: List[float]) -> int:
        """計算最佳入場點位"""
        if not forecast_prices:
            return 1
        
        if trend['direction'] == 'uptrend':
            best_bar = forecast_prices.index(min(forecast_prices)) + 1
        else:
            best_bar = forecast_prices.index(max(forecast_prices)) + 1
        
        return best_bar
    
    def generate_recommendation(self, trend: dict, best_entry_bar: int, 
                               price_extremes: dict, forecast_prices: List[float]) -> str:
        """生成詳細的交易建議"""
        if best_entry_bar > len(forecast_prices):
            return "無法需議"
        
        entry_price = forecast_prices[best_entry_bar - 1]
        
        if trend['direction'] == 'uptrend':
            profit_target = price_extremes.get('highest_price', entry_price)
            stop_loss = price_extremes.get('lowest_price', entry_price) * 0.99
            profit_pct = (profit_target - entry_price) / entry_price * 100 if entry_price > 0 else 0
            risk_pct = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
            
            recommendation = f"""
交易信號：強勢多頭
趨勢分析：{trend.get('description', '')}

最優策略：
- 在第 {best_entry_bar} 根K棒進行開多單
- 入場價格：${entry_price:.8f}
- 止盈目標：${profit_target:.8f}（潛在收益：+{profit_pct:.2f}%）
- 止損位置：${stop_loss:.8f}（控制風險：{risk_pct:.2f}%）
- 風險回報比：1:{(profit_pct/max(risk_pct, 0.01)):.2f}
            """
        else:
            profit_target = price_extremes.get('lowest_price', entry_price)
            stop_loss = price_extremes.get('highest_price', entry_price) * 1.01
            profit_pct = (entry_price - profit_target) / entry_price * 100 if entry_price > 0 else 0
            risk_pct = (stop_loss - entry_price) / entry_price * 100 if entry_price > 0 else 0
            
            recommendation = f"""
交易信號：強勢空頭
趨勢分析：{trend.get('description', '')}

最優策略：
- 在第 {best_entry_bar} 根K棒進行開空單
- 入場價格：${entry_price:.8f}
- 止盈目標：${profit_target:.8f}（潛在收益：+{profit_pct:.2f}%）
- 止損位置：${stop_loss:.8f}（控制風險：{risk_pct:.2f}%）
- 風險回報比：1:{(profit_pct/max(risk_pct, 0.01)):.2f}
            """
        
        return recommendation.strip()
    
    def analyze(self, current_price: float, historical_prices: List[float],
               forecast_prices: List[float], symbol: str, timeframe: str) -> dict:
        """執行完整的市場分析"""
        trend = self.analyze_trend(historical_prices)
        price_extremes = self.find_price_extremes(forecast_prices)
        best_entry_bar = self.calculate_best_entry(trend, forecast_prices)
        recommendation = self.generate_recommendation(trend, best_entry_bar, price_extremes, forecast_prices)
        
        return {
            'trend': trend,
            'best_entry_bar': best_entry_bar,
            'price_extremes': price_extremes,
            'recommendation': recommendation
        }

# ============================================================================
# 粗推到 app.py: 在其他端點之例添加 此 POST 端點
# ============================================================================

# 在你的 V5 app.py 中，找到 @app.post("/predict-v5") 或非最後一個 POST 端點
# 然後在之後添加下面的程式碼:

"""
# 粗推 market_analyzer 全吲 你的初始化段院
# (app = FastAPI(...) 之後)
market_analyzer = MarketAnalyzer()

# 粗推 @app.post("/market-analysis") 端點
@app.post("/market-analysis")
async def market_analysis(request: dict) -> dict:
    """市場分析 - 趨勢判斷和最佳入場點計算"""
    
    try:
        symbol = request.get('symbol', 'BTC')
        timeframe = request.get('timeframe', '1d')
        use_binance = request.get('use_binance', False)
        
        logger.info(f"[Market Analysis] Analyzing {symbol} ({timeframe})")
        
        # 符号映射
        symbol_map = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
            'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
            'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
            'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
            'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
        }
        
        binance_symbol = symbol_map.get(symbol, symbol + 'USDT')
        tf = '1d' if timeframe == '1d' else '1h'
        
        # 獲取數據 (30根K棒: 20感歷伸 + 10予測)
        # 需要你的 data_fetcher 實例
        all_klines = await data_fetcher.fetch_klines(
            binance_symbol,
            timeframe=tf,
            limit=30
        )
        
        if len(all_klines) < 20:
            return {
                "error": f"Insufficient data: got {len(all_klines)} klines",
                "status": "failed"
            }
        
        historical_klines = all_klines[:20]
        forecast_klines = all_klines[20:30]
        
        historical_prices = [k['close'] for k in historical_klines]
        forecast_prices = [k['close'] for k in forecast_klines]
        current_price = historical_klines[-1]['close']
        
        # 執行市場分析
        analysis = market_analyzer.analyze(
            current_price=current_price,
            historical_prices=historical_prices,
            forecast_prices=forecast_prices,
            symbol=symbol,
            timeframe=timeframe
        )
        
        logger.info(f"[Market Analysis] Success: {symbol} -> {analysis['trend']['direction']}")
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend": analysis['trend'],
            "best_entry_bar": analysis['best_entry_bar'],
            "price_extremes": analysis['price_extremes'],
            "forecast_prices": forecast_prices,
            "recommendation": analysis['recommendation'],
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"[Market Analysis] Error: {e}")
        return {
            "error": str(e),
            "status": "error"
        }
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║               V5 MARKET ANALYSIS PATCH - 整合指示                       ║
╚════════════════════════════════════════════════════════════════════════════╝

請推錄本文件中的程式碼到你的 app.py！

步驥：

1. 打開你的 V5 app.py

2. 在文件頂端的 import 之後，貼上：
   - 整個 "MarketAnalysisRequest" 類；
   - 整個 "MarketAnalyzer" 類；

3. 在 "if __name__ == '__main__'" 前（例如 app 初始化之後），添加：
   - market_analyzer = MarketAnalyzer()

4. 在最後一個 @app.post 端點之後，貼負上片【粗推到 app.py: 在其他端點之例添加】中的整個 POST 端點。

5. 重新啟動 API

6. 測試:
   curl -X POST http://localhost:8001/market-analysis \\
     -H "Content-Type: application/json" \\
     -d '{
       "symbol": "BTC",
       "timeframe": "1d",
       "use_binance": false
     }'

需要更多幫助? 參考 INTEGRATION_GUIDE.md
""")
