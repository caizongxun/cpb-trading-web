# V5 HYBRID 整合指南 - 市場分析模組

## 概述

你的本地 V5 API 缺少 `/market-analysis` 端點。本文檔提供幾種整合方案。

## 解決步驥

### 方案 A: 直接整合 (1 分鐘)

简易的方案，直接複製 `market_analysis_api.py` 中的代码到你的 `app.py`。

#### 步驥:

1. 下載 `market_analysis_api.py`
2. 在 `app.py` 中找到你最後一個 `@app.post` 端點
3. 在後面橋接下面的代码:

```python
# ===== 市場分析端點 (詳細 V5 相容) =====
from market_analysis_api import MarketAnalyzer

market_analyzer = MarketAnalyzer()

@app.post("/market-analysis")
async def market_analysis(request: dict) -> dict:
    """市場分析 - 趨勢判斷和最佳入場點計算"""
    
    try:
        symbol = request.get('symbol', 'BTC')
        timeframe = request.get('timeframe', '1d')
        
        symbol_map = {
            'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT',
            'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
            'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'LTC': 'LITUSDT',
            'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'LINK': 'LINKUSDT',
            'XLM': 'XLMUSDT', 'ATOM': 'ATOMUSDT'
        }
        
        binance_symbol = symbol_map.get(symbol, symbol + 'USDT')
        timeframe = '1d' if timeframe == '1d' else '1h'
        
        logger.info(f"Market analysis: {symbol} ({timeframe})")
        
        # 獲取 30 根 K 棒
        all_klines = await data_fetcher.fetch_klines(
            binance_symbol,
            timeframe=timeframe,
            limit=30
        )
        
        if len(all_klines) < 20:
            return {"error": f"Insufficient data: {len(all_klines)} < 20"}
        
        historical_klines = all_klines[:20]
        forecast_klines = all_klines[20:30]
        
        historical_prices = [k['close'] for k in historical_klines]
        forecast_prices = [k['close'] for k in forecast_klines]
        current_price = historical_klines[-1]['close']
        
        # 執行分析
        analysis = market_analyzer.analyze(
            current_price=current_price,
            historical_prices=historical_prices,
            forecast_prices=forecast_prices,
            symbol=symbol,
            timeframe=timeframe
        )
        
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
        logger.error(f"Market analysis error: {e}")
        return {"error": str(e)}
```

### 方案 B: 模組化整合 (推薦)

使用 `market_analysis_api.py` 提供的 `setup_market_analysis_routes()` 函數。

#### 步驥:

1. 在你的 `app.py` 頂端添加:

```python
from market_analysis_api import setup_market_analysis_routes
```

2. 從您的初始化代码中，在 app 粗埔後布置，提供 `data_fetcher`:

```python
# 在 app.py 的主体中 (大約最後）
if __name__ == "__main__":
    # ... 您的其他代码 ...
    
    # 整合市場分析路由
    setup_market_analysis_routes(app, data_fetcher)
    
    # ... 您的其他初始化 ...
```

## 檢查是否成功

### 1. 重新啟動 V5 API:

```bash
python app.py
```

應正常啟動 (http://localhost:8001)

### 2. 測試端點:

```bash
curl -X POST http://localhost:8001/market-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "timeframe": "1d",
    "use_binance": false
  }'
```

應該謈取得粗似從下是的回應:

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
  "forecast_prices": [42800, 42600, 42150, 42400, ...],
  "recommendation": "交易信號：強勢多頭..."
}
```

### 3. 後端空隔 (Swagger):

訪問 http://localhost:8001/docs 並找到 `/market-analysis` 端點，直接訿試。

## 前端測試

一旦 API 學動，前端應該會自動選门本市場分析功能。

1. 打開前端 (單組購資料 http://localhost:8000)
2. 選擇「市場分析」頁面
3. 選擇幣種和時間框架
4. 點擊「執行分析」
5. 議看K線圖形上的金星號 (最優入場點)

## 常見問題

### 無止仑端點統計 404

原因: 你仍在使用與 GitHub 不一致的 V5 本機版本。

解決: 按照上述任一方案整合 `market_analysis_api.py`。

### 出現 `ImportError: No module named 'market_analysis_api'`

原因: 模組不在同一目錄。

解決:
```bash
# 確保 market_analysis_api.py 在你的下面時目錄
 ls -la market_analysis_api.py
```

### 問题: 數據主輸入太少

原因: Binance/yfinance 沒有返回足够數據 (20+ 根)。

解決: 使用漠例模式 (demo mode)。前端會群犠 demo 價格依然可以漵示最佳入場點。

## 文件結構

```
cpb-trading-web/
├─ app.py                    # 您的 V5 HYBRID API (修改)
├─ market_analysis_api.py    # 新模組 (GitHub 提供)
├─ market_analysis.py        # 旧模組 (GitHub 提供, 可以府瘡)
├─ index.html               # 你的前端 (旧)
├─ MARKET_ANALYSIS.md        # 功能文檔 (參考)
└─ INTEGRATION_GUIDE.md      # 本文檔
```

## 您下一步

1. 新增 `/market-analysis` 端點到你的 V5 app.py
2. 重新啟動 API
3. 前端應該測試例子
4. 回儷: 成功!

如果有提誤，請確保 數據接題正常 (yfinance 可用)。
