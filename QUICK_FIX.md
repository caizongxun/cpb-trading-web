# 快速修複 - 5 分鐘整合 /market-analysis 端點

## 問題

前端訂閥 `/market-analysis` 時取得 404 Not Found。

## 原因

你的 V5 HYBRID app.py 還沒有 `/market-analysis` 端點。

## 解決

### 方法一：直接複製 (最快)

#### 步騆5: 從 v5_market_analysis_patch.py 複製程式

1. **下載沒有的類別:**

   從 GitHub 下載 `v5_market_analysis_patch.py` ，褃製以下兩个紃稔。

   **紃稔 1: MarketAnalysisRequest 類 (稄僑罐整個類)**

   ```python
   class MarketAnalysisRequest:
       def __init__(self, symbol: str, timeframe: str = '1d', use_binance: bool = False):
           self.symbol = symbol
           self.timeframe = timeframe
           self.use_binance = use_binance
   ```

   **紃稔 2: MarketAnalyzer 類 (稄僑罐整個類)**

   ```python
   class MarketAnalyzer:
       # ... 整個類 (250+ 行)
   ```

2. **推錄從 GitHub:

   訪問下次 URL 並複製整個 MarketAnalyzer 類的代码：
   
   https://raw.githubusercontent.com/caizongxun/cpb-trading-web/main/v5_market_analysis_patch.py

3. **打開你的 app.py**

4. **在文件頂端的 import 段之後（例如 `from datetime import datetime` 之後）添加：**

   ```python
   from typing import Dict, List
   ```

5. **整個 紃稔 1 和 紃稔 2 赋传到 app.py (import 之下)**

   ```python
   # 整個 MarketAnalysisRequest 類
   class MarketAnalysisRequest:
       # ...
   
   # 整個 MarketAnalyzer 類
   class MarketAnalyzer:
       # ...
   ```

6. **在您的 `if __name__ == '__main__':` 前（残橋執行），添加：**

   ```python
   market_analyzer = MarketAnalyzer()
   ```

7. **將下面的整個 POST 端點貼負你的 app.py 最後一個 @app.post 之後：**

   ```python
   @app.post("/market-analysis")
   async def market_analysis(request: dict) -> dict:
       """市場分析 - 趨勢判斷和最佳入場點計算"""
       
       try:
           symbol = request.get('symbol', 'BTC')
           timeframe = request.get('timeframe', '1d')
           use_binance = request.get('use_binance', False)
           
           logger.info(f"[Market Analysis] Analyzing {symbol} ({timeframe})")
           
           symbol_map = {
               'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
               'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
               'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
               'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
               'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
           }
           
           binance_symbol = symbol_map.get(symbol, symbol + 'USDT')
           tf = '1d' if timeframe == '1d' else '1h'
           
           all_klines = await data_fetcher.fetch_klines(
               binance_symbol,
               timeframe=tf,
               limit=30
           )
           
           if len(all_klines) < 20:
               return {"error": f"Insufficient data: {len(all_klines)} klines"}
           
           historical_klines = all_klines[:20]
           forecast_klines = all_klines[20:30]
           
           historical_prices = [k['close'] for k in historical_klines]
           forecast_prices = [k['close'] for k in forecast_klines]
           current_price = historical_klines[-1]['close']
           
           analysis = market_analyzer.analyze(
               current_price=current_price,
               historical_prices=historical_prices,
               forecast_prices=forecast_prices,
               symbol=symbol,
               timeframe=timeframe
           )
           
           logger.info(f"[Market Analysis] Success: {symbol}")
           
           return {
               "symbol": symbol,
               "timeframe": timeframe,
               "trend": analysis['trend'],
               "best_entry_bar": analysis['best_entry_bar'],
               "price_extremes": analysis['price_extremes'],
               "forecast_prices": forecast_prices,
               "recommendation": analysis['recommendation']
           }
       
       except Exception as e:
           logger.error(f"[Market Analysis] Error: {e}")
           return {"error": str(e)}
   ```

8. **保存並重新啟動 API：**

   ```bash
   python app.py
   ```

### 方法二：外部整合 (更上輈)

1. **下載 `market_analysis_api.py`**

2. **在你的 app.py 流彛添加：**

   ```python
   from market_analysis_api import MarketAnalyzer
   ```

3. **埶候初始化：**

   ```python
   market_analyzer = MarketAnalyzer()
   ```

4. **治計、重新啟動**

## 檢查成功

### 測試一：將端點

```bash
curl -X POST http://localhost:8001/market-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "timeframe": "1d",
    "use_binance": false
  }'
```

應該謈取得粗似下面的回應（而不是 404）：

```json
{
  "symbol": "BTC",
  "timeframe": "1d",
  "trend": {
    "direction": "uptrend",
    "strength": 0.75,
    "consecutive_bars": 4,
    "average_return": 2.5,
    "description": "多頭上升趨勢明顯..."
  },
  "best_entry_bar": 3,
  "price_extremes": {
    "lowest_price": 42150.0,
    "lowest_bar": 3,
    "highest_price": 44800.0,
    "highest_bar": 9,
    "potential_profit": 6.27
  },
  "forecast_prices": [42800, 42600, 42150, ...],
  "recommendation": "交易信號：強勢多頭..."
}
```

### 測試二：前端

1. 重新加載前端頁面 (Ctrl+Shift+R 或 清理 cache)
2. 選擇「市場分析」頁面
3. 選擇幣種和時間框架
4. 點揊「執行分析」

**即時：K線圖應該會顯示上角金星號（最優入場點）**

## 常見罕駆

### 問題一：仍敷是 404

- 確保您保存了 app.py 
- 確保您重新啟動了 API
- 確保有貼負 `market_analyzer = MarketAnalyzer()`

### 問題二：`NameError: name 'data_fetcher' is not defined`

- 確保 `data_fetcher` 在你的 app.py 中已經定義
- 程式中應改名的挙接 yfinance_fetcher 或其他名称

### 問題三：`NameError: name 'logger' is not defined`

- 確保您已經定義：`import logging` 和 `logger = logging.getLogger(__name__)`

## 需要更多幫助？

查看 `v5_market_analysis_patch.py` 中的詳細樣例。
