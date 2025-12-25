import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class TrendAnalysis:
    """趨勢分析結果"""
    direction: str  # 'uptrend' or 'downtrend'
    strength: float  # 0-1
    consecutive_bars: int  # 連續上升或下降的根數
    average_return: float  # 平均漲幅
    description: str


@dataclass
class PriceExtremes:
    """價格極值分析"""
    lowest_price: float
    lowest_bar: int
    highest_price: float
    highest_bar: int
    potential_profit: float  # 從最低到最高的利潤百分比


@dataclass
class MarketAnalysisResult:
    """市場分析完整結果"""
    symbol: str
    timeframe: str
    trend: TrendAnalysis
    best_entry_bar: int  # 最佳入場的K棒位置 (1-10)
    price_extremes: PriceExtremes
    forecast_prices: List[float]  # 未來10根K棒的預測價格
    recommendation: str


class MarketAnalyzer:
    """市場分析引擎"""
    
    def __init__(self):
        self.min_trend_strength = 0.3  # 最小趨勢強度判定
        self.min_consecutive = 3  # 最少連續根數
    
    def analyze_trend(self, historical_prices: List[float], forecast_prices: List[float]) -> TrendAnalysis:
        """
        分析趨勢方向和強度
        """
        # 分析最近的K線方向
        recent_count = min(5, len(historical_prices))
        recent_prices = historical_prices[-recent_count:]
        
        # 計算上升和下降的根數
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
        
        # 計算趨勢強度
        total = up_count + down_count
        strength = up_count / total if total > 0 else 0.5
        
        # 判定趨勢方向
        if strength > 0.5:
            direction = 'uptrend'
            consecutive = max_consecutive_up
            avg_return = np.mean(np.diff(recent_prices)) / recent_prices[0] if len(recent_prices) > 1 else 0
        else:
            direction = 'downtrend'
            consecutive = max_consecutive_down
            avg_return = np.mean(np.diff(recent_prices)) / recent_prices[0] if len(recent_prices) > 1 else 0
        
        # 生成描述
        if strength > 0.7:
            strength_desc = '強勢'
        elif strength > 0.5:
            strength_desc = '中等'
        else:
            strength_desc = '弱'
        
        trend_name = '多頭上升' if direction == 'uptrend' else '空頭下跌'
        description = f"{trend_name}趨勢明顯，{strength_desc}程度，最近{consecutive}根K線連續{('上升' if direction == 'uptrend' else '下跌')}"
        
        return TrendAnalysis(
            direction=direction,
            strength=max(min(strength, 1.0), 0.0),
            consecutive_bars=consecutive,
            average_return=avg_return,
            description=description
        )
    
    def find_price_extremes(self, forecast_prices: List[float]) -> PriceExtremes:
        """
        尋找未來10根K棒中的最高和最低價格
        """
        lowest_price = min(forecast_prices)
        highest_price = max(forecast_prices)
        
        lowest_bar = forecast_prices.index(lowest_price) + 1
        highest_bar = forecast_prices.index(highest_price) + 1
        
        potential_profit = (highest_price - lowest_price) / lowest_price
        
        return PriceExtremes(
            lowest_price=lowest_price,
            lowest_bar=lowest_bar,
            highest_price=highest_price,
            highest_bar=highest_bar,
            potential_profit=potential_profit
        )
    
    def calculate_best_entry(self, 
                            trend: TrendAnalysis,
                            current_price: float,
                            forecast_prices: List[float]) -> Tuple[int, str]:
        """
        計算最佳入場點位
        
        多頭：尋找未來10根中的最低點
        空頭：尋找未來10根中的最高點
        """
        if trend.direction == 'uptrend':
            # 多頭：找最低點入場
            best_bar = forecast_prices.index(min(forecast_prices)) + 1
            reason = f"在第{best_bar}根K棒（最低點）開多單，之後有上升空間"
        else:
            # 空頭：找最高點入場
            best_bar = forecast_prices.index(max(forecast_prices)) + 1
            reason = f"在第{best_bar}根K棒（最高點）開空單，之後有下跌空間"
        
        return best_bar, reason
    
    def generate_recommendation(self,
                               trend: TrendAnalysis,
                               best_entry_bar: int,
                               price_extremes: PriceExtremes,
                               forecast_prices: List[float]) -> str:
        """
        生成詳細的交易建議
        """
        entry_price = forecast_prices[best_entry_bar - 1]
        
        if trend.direction == 'uptrend':
            # 多頭建議
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

注意事項：
- 趨勢強度強勢，建議跟隨多頭方向
- 在相對低點進場，利潤空間較大
- 設置止損保護資金安全
            """
        else:
            # 空頭建議
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

注意事項：
- 趨勢強度強勢，建議跟隨空頭方向
- 在相對高點進場，利潤空間較大
- 設置止損保護資金安全
            """
        
        return recommendation.strip()
    
    def analyze(self,
               current_price: float,
               historical_prices: List[float],
               forecast_prices: List[float],
               symbol: str,
               timeframe: str) -> MarketAnalysisResult:
        """
        執行完整的市場分析
        """
        # 分析趨勢
        trend = self.analyze_trend(historical_prices, forecast_prices)
        
        # 找出價格極值
        price_extremes = self.find_price_extremes(forecast_prices)
        
        # 計算最佳入場點
        best_entry_bar, _ = self.calculate_best_entry(trend, current_price, forecast_prices)
        
        # 生成建議
        recommendation = self.generate_recommendation(
            trend,
            best_entry_bar,
            price_extremes,
            forecast_prices
        )
        
        return MarketAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            trend=trend,
            best_entry_bar=best_entry_bar,
            price_extremes=price_extremes,
            forecast_prices=forecast_prices,
            recommendation=recommendation
        )
