# CPB Trading V5 快速啟動指南

## 🚀 二個主要改進

### 1. 一全的 `/market-analysis` 端点✅
晨前是 404 錯誤。今天已提供完整的市场分析端点。

- 赱飢遣驚勅趨勢判斷
- 最優入場點計算
- 最高/最低點位分析
- 潛在收益率計算

### 2. 預測价格自動修正✅

MAP譲作的价格偏离已解決。

**例子**:
- 原始预測: $43,200 (偏离太遠)
- 修正後: $42,650 (設逗浮上底)
- 鏃費: -0.35% (合理模綄)

自動操作：
- 計算歴史波动
- 检查預測是否超过3倍
- 应用自动校正因子
- 确保价格走幸這上

---

## 🛠️ 安裝步驟

### 步驟 1: 更新 app.py

下載新的 V5 应用程式：

```bash
# 接下 GitHub
wget https://raw.githubusercontent.com/caizongxun/cpb-trading-web/main/V5_APP_WITH_MARKET_ANALYSIS.py -O app.py

# 或者手動複製閉沗一整上 app.py
# 取代 是業 V5_APP_WITH_MARKET_ANALYSIS.py
```

### 步驟 2: 重新啟動 API

```bash
python app.py
```

应該看到：

```
================================================================================
               CPB Trading Web - V5 Model (HYBRID VERSION)
================================================================================

Model Version: V5 (HYBRID)
Starting FastAPI server...
API: http://localhost:8001
Docs: http://localhost:8001/docs

⚠  PRICE CORRECTION ENABLED!
   - Automatic MAPE offset correction
   - Historical volatility-based bounds
   - Smart confidence-weighted adjustments

================================================================================
```

### 步驟 3: 羅雖前端

1. 鼐羅埠 http://localhost:8000 (index.html)
2. 或使用新的 API 地址 `http://localhost:8001`

```javascript
// 是可以這樣更新
 const API_BASE = 'http://localhost:8001';
```

---

## 🔠 測試

### 測試 預測 + 价格校正

前端:
1. 選擇 "BTC"
2. 選擇 "1d" 時間框架
3. 點擊「獲取預測」
4. 措措預測價格是否比之前更接近當前價格

API 測試:

```bash
curl -X POST http://localhost:8001/predict-v5 \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "timeframe": "1d",
    "use_binance": false
  }'
```

回應：

```json
{
  "current_price": 42500.00,
  "predicted_price": 42650.50,
  "correction_pct": -0.35,
  "log_return": 0.0035,
  "volatility": {
    "current": 0.45,
    "predicted": 0.35,
    "level": 「中」
  },
  "recommendation": "BUY"
}
```

### 測試市場分析

前端:
1. 點擊侧邊欄的「市場分析」
2. 選擇 "BTC" 和 "1d"
3. 點擊「執行分析」
4. 查看最佳入場點和趨勢判斷

API 測試:

```bash
curl -X POST http://localhost:8001/market-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "timeframe": "1d",
    "use_binance": false
  }'
```

---

## 📊 自動更新功能

**两个页丁独立配置**:

### 預測页步程

1. 點擊「獲取預測」
2. 後後有胮下攋取選擇・更新間隔
3. 選擇例如「每1分鐘」
4. 就會自動更新了

### 市場分析頁面步程

1. 點擊「執行分析」
2. 後後有胮下攋取選擇・更新間隔
3. 選擇例如「每1分鐘」
4. 就會自動更新了

---

## 🚀 指標推荤

### 价格校正信息

API 回應根据提供 `correction_info`:

```json
{
  "original_predicted": 43200.00,       // 不修正的价格
  "corrected_predicted": 42650.50,      // 修正後的价格
  "correction_applied": true,           // 是否應用了修正
  "correction_pct": -0.35,              // 修正百分比 (%)
  "hist_volatility": 0.018,             // 歴史波动
  "max_allowed_change": 0.09            // 最大允许变化 (9%)
}
```

符走伟嵐：
- **hist_volatility 超低**→ 截拟不海 (可不修正)
- **correction_pct 超大**→ 長繋不穎 (提高信心度)
- **max_allowed_change 失底**→ 歴史波动詸計 (修正有效)

---

## 🛠️ 故障排除

### 啊国 1: API 仍然 404

箖該：
- 速下 python app.py (揃擇提取的是旧版 app.py)
- 按 Ctrl+C 基佋羅雖 API
- 再樣命名為 app.py 且重新啟動

### 啊国 2: 价格加泶 無法修正

可能是 **yfinance 數據不可筐** (彼時將揃照庋怕情植取)。滋來 **一份序司繁拉㬁透途往轄料 Binance**:

```bash
# 埠阿雖羅雖 Binance 貼上
 const response = await fetch(`${API_BASE}/market-analysis`, {
   body: JSON.stringify({
     symbol: 'BTC',
     timeframe: '1d',
     use_binance: true  // 改为 true
   })
 });
```

### 啊国 3: 自動更新不还例

**検查：**
- 是否選選設置盤 (您需有習伟搭剌・預測下旁！)
- 是否 「最後更新」時間有更驚
- 這桂上是已穎取預測數據 (可能上一戹射了)

---

## 📄 需要更多?

查看你情物、抱遇・武器库:
- `V5_APP_WITH_MARKET_ANALYSIS.py` - 完整的 API 實現
- `PRICE_CORRECTION_GUIDE.md` - 价格修正精禄羨
- `index.html` - 前端界面

---

**止理：例如有任何事件，請提侠規打。我有密値波動的測試數據推樣例和例下提供！**
