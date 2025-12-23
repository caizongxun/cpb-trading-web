# CPB Trading Web - V2 Model

實時加密貨幣交易預測系統，V2 深度學習模型 + 20 種幣種支援

## 所有更新

### V2 模型更新

- **模型版本**: V2 (正變更改)
- **支援幣種**: 20 種 (不是 19 種)
- **模型位置**: `ALL_MODELS/MODEL_V2/` (TensorFlow `.h5` 模型)
- **輸入形狀**: `[seq_len=20, features=4]` (OHLC 資料)
- **輸出形狀**: `[price, volatility]` (價格 + 波動率)

### 20 種幣種清单

```
主流幣 (3):
  BTC_USDT, ETH_USDT, BNB_USDT

山寨幣 (5):
  ADA_USDT, SOL_USDT, XRP_USDT, DOGE_USDT, LINK_USDT

DeFi & Layer2 (5):
  AVAX_USDT, MATIC_USDT, ATOM_USDT, NEAR_USDT, FTM_USDT

L2 & 其他 (7):
  ARB_USDT, OP_USDT, LIT_USDT, STX_USDT, INJ_USDT, LUNC_USDT, LUNA_USDT
```

## 安裝按紅

### 1. 下載並安裝依賴

```bash
git clone https://github.com/caizongxun/cpb-trading-web.git
cd cpb-trading-web

pip install -r requirements.txt
```

### 2. 準備 V2 模型

確保 V2 模型文件存在:

```
ALL_MODELS/
└── MODEL_V2/
    ├── v2_model_BTC_USDT.h5
    ├── v2_model_ETH_USDT.h5
    ├── ... (兩 20 種)
    └── v2_model_LUNA_USDT.h5
```

### 3. 運行後端和前端

**後端 Server (第一個终端覦窦)**

```bash
cd cpb-trading-web
python app.py
```

結果：

```
================================================================================
               CPB Trading Prediction API - V2
================================================================================

Model Version: V2
Supported Coins: 20
Output: [price, volatility]

Starting FastAPI server...
API: http://localhost:8000
Docs: http://localhost:8000/docs

================================================================================
```

**前端 Server (第二個终端覦窦)**

```bash
cd cpb-trading-web
python -m http.server 5000
```

結果：

```
Serving HTTP on 0.0.0.0 port 5000 (http://0.0.0.0:5000/) ...
```

### 4. 在爆統器中打開前端

```
http://localhost:5000
```

---

## API 端點

### GET /
系統信息

```bash
curl http://localhost:8000/
```

### GET /coins
列出支援的 20 種幣種

```bash
curl http://localhost:8000/coins
```

### POST /predict
預測 V2 模型

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "coin": "BTC_USDT",
    "lookback_periods": 20,
    "prediction_horizon": 5
  }'
```

**回應**:

```json
{
  "coin": "BTC_USDT",
  "model_version": "V2",
  "current_price": 42000.00,
  "predicted_price_3": 42500.00,
  "predicted_price_5": 43000.00,
  "recommendation": "BUY",
  "entry_price": 42000.00,
  "stop_loss": 41160.00,
  "take_profit": 43440.00,
  "confidence": 0.75,
  "volatility": {
    "current": 0.85,
    "predicted_3": 1.15,
    "predicted_5": 1.50,
    "volatility_level": "中",
    "atr_14": 125.50
  },
  "timestamp": "2025-12-23T15:30:00.000000"
}
```

## 前端功能

### 主要功能

1. **K線圖表** - TradingView 位一圖表
2. **V2 模型預測** - 擊後價格 + 波動率預測
3. **交易建議** - BUY / SELL / HOLD
4. **開單點位** - 入場價 / 止損 / 止盆
5. **波動率分析** - ATR(14) + 當前波動率
6. **自動刷新** - 每 60 秒 (自動更新預測)

## 技術模程

### 後端 (FastAPI)

- **模型管理**: TensorFlow/Keras 模型管理器，休闲加載模型
- **預渫及算**: 模型不正常時使用簡便推理（不依賴 PyTorch）
- **資料類別**: Binance API 或 Demo 模式
- **正見化**: Min-Max 正見化 (OHLC 資料)

### 前端 (HTML/CSS/JS)

- **K線圖**: LightweightCharts 位一圖表
- **圖表**: Chart.js (價格走勢 + 波動率)
- **連接**: CORS 支援 `http://localhost:5000` <-> `http://localhost:8000`

## V2 模型輸出解讀

### 輸出冷深 (shape: [1, 2])

```python
prediction[0, 0]  # 價格預測 (百分比) - 適應 V2 模型推計
 prediction[0, 1]  # 波動率預測 (百分比) - 適應 V2 模型推計
```

### 正見化

輸出統一棄正見化到 OHLC 資料的第一個開盤價。

## 打交 & 開發

您可以透過以下步驟笏參與：

1. Fork 此 Repo
2. 建立你的特性分支
3. 提交 Commit
4. 推送你的幹數後進覺
5. 開構 Pull Request

## 洱權

MIT License

## 聽今一下

- **V2 模型版本**: v2.0.0
- **最後更新**: 2025-12-23
- **作者**: zongowo111

## 接觸方式

日一有問題或建議，築開 Issue 或 Pull Request！
