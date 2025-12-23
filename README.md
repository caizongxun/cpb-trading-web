# CPB Trading Web - 加密貨幣交易信號推薦系統

實時加密貨幣交易信號預測和開單位置推薦。

## 功能特點

- **實時數據**: 從 Binance 實時抓取 K 棒數據
- **AI 預測**: 基於 HuggingFace 訓練的 PyTorch 模型
- **多幣種支持**: 同時支持 19 個主流幣種
- **完整推薦**: 提供 BUY/SELL/HOLD 信號 + 開單點位 + 止損止盈
- **信心指數**: 顯示預測的信心度（0-100%）
- **批量預測**: 一鍵預測全部幣種

## 架構

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────┐
│   index.html    │────────▶│   FastAPI (8000) │────────▶│  HuggingFace │
│   (前端)        │         │   (後端)         │         │  (模型)      │
└─────────────────┘         └──────────────────┘         └──────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Binance API     │
                            │  (實時行情)      │
                            └──────────────────┘
```

## 快速開始

### 1. 克隆倉庫

```bash
git clone https://github.com/caizongxun/cpb-trading-web.git
cd cpb-trading-web
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 設置環境變數

複製並編輯 `.env.example`：

```bash
cp .env.example .env
```

編輯 `.env` 文件：

```
HF_TOKEN=hf_你的token
HF_USERNAME=你的username
```

### 4. 運行後端

**在 PyCharm 或命令行**：

```bash
python app.py
```

或：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

看到這個信息表示成功：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### 5. 打開前端

直接在瀏覽器打開 `index.html` 或使用簡單的 HTTP 伺服器：

```bash
# Python 3
python -m http.server 5000

# 或 Node.js
npx http-server
```

然後打開：http://localhost:5000

## 使用方法

### 單幣種預測

1. 從下拉菜單選擇幣種
2. 設置看回週期（默認 20）
3. 點擊「單幣種預測」按鈕

### 批量預測

點擊「批量預測全部」，會同時預測所有支持的 19 個幣種。

## API 端點

### GET `/coins`

列出所有支持的幣種

```bash
curl http://localhost:8000/coins
```

### POST `/predict`

預測單個幣種

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "coin": "BTCUSDT",
    "lookback_periods": 20,
    "prediction_horizon": 5
  }'
```

**返回示例**:

```json
{
  "coin": "BTCUSDT",
  "timestamp": "2025-12-23T10:00:00",
  "current_price": 42500.50,
  "predicted_price_3": 43000.00,
  "predicted_price_5": 43500.00,
  "recommendation": "BUY",
  "entry_price": 42500.50,
  "stop_loss": 41650.49,
  "take_profit": 44370.00,
  "confidence": 0.75,
  "klines": [...]
}
```

### POST `/predict-batch`

批量預測多個幣種

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '["BTCUSDT", "ETHUSDT", "BNBUSDT"]'
```

### GET `/health`

健康檢查

```bash
curl http://localhost:8000/health
```

## 支持的幣種

1. BTCUSDT - Bitcoin
2. ETHUSDT - Ethereum
3. BNBUSDT - BNB
4. ADAUSDT - Cardano
5. SOLUSDT - Solana
6. XRPUSDT - Ripple
7. DOGEUSDT - Dogecoin
8. LTCUSDT - Litecoin
9. LINKUSDT - Chainlink
10. UNIUSDT - Uniswap
11. AVAXUSDT - Avalanche
12. ATOMUSDT - Cosmos
13. VETUSDT - VeChain
14. GRTUSDT - The Graph
15. AXSUSDT - Axie Infinity
16. BCHUSDT - Bitcoin Cash
17. MANAUSDT - Decentraland
18. SANDUSDT - Sandbox
19. XLMUSDT - Stellar

## 模型來源

所有模型都託管在 HuggingFace:

- **Repo**: `zongowo111/cpbmodel`
- **模型類型**: PyTorch 1h 時框
- **訓練數據**: 歷史 OHLCV 數據

## 推薦信號說明

### BUY (買入)
- 模型預測未來 3-5 根 K 棒上漲
- 開單點位：當前價格
- 止損：-2%
- 止盈：預測價格上方 2%

### SELL (賣出)
- 模型預測未來 3-5 根 K 棒下跌
- 開單點位：當前價格
- 止損：+2%
- 止盈：預測價格下方 2%

### HOLD (持有)
- 模型不確定或信號不明確
- 建議暫不操作

## 故障排除

### API 連接失敗

```
API 未連線，請確保 FastAPI 伺服器正在運行
```

解決：確保後端已啟動（`python app.py`）

### 模型加載失敗

```
Model load failed: ...
```

解決：
1. 檢查 `HF_TOKEN` 是否正確設置
2. 檢查網絡連接
3. 檢查模型是否在 HuggingFace 上存在

### 數據抓取失敗

```
Data fetch failed: ...
```

解決：
1. 檢查網絡連接
2. 檢查 Binance API 是否可用
3. 檢查幣種名稱是否正確

## PyCharm 設置

### 1. 打開項目

File → Open → 選擇 `cpb-trading-web` 文件夾

### 2. 配置 Python 環境

PyCharm → Preferences → Project → Python Interpreter → Add Interpreter → Add Local

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

### 4. 創建運行配置

- 點擊 Run → Edit Configurations
- 點擊 + 添加新配置
- 選擇 Python
- Script path: 選擇 `app.py`
- 點擊 OK
- 點擊運行或按 Shift+F10

### 5. 在 PyCharm 的終端中運行

```bash
python app.py
```

## 開發說明

### 修改預測邏輯

編輯 `app.py` 中的 `ModelManager._forward_simple_model()` 方法。

### 自定義推薦規則

編輯 `/predict` 端點中的推薦邏輯部分。

### 添加新幣種

1. 確認 HuggingFace 上有該幣種的模型
2. 將幣種名稱添加到 `SUPPORTED_COINS` 列表
3. 在前端 HTML 的 `select` 中添加選項

## 性能優化

- 模型會在第一次請求時加載並緩存
- 支持 CUDA GPU 加速（自動檢測）
- 異步 I/O 防止阻塞
- 批量預測優化

## 安全性注意

- 不要將 `HF_TOKEN` 提交到 git
- 使用 `.env` 文件管理敏感信息
- 在生產環境中使用 HTTPS
- 考慮添加 API 密鑰認證

## 貢獻

Issues 和 Pull Requests 歡迎！

## 許可証

MIT License

## 聯繫方式

- GitHub: [@caizongxun](https://github.com/caizongxun)
- HuggingFace: [@zongowo111](https://huggingface.co/zongowo111)

---

**免責聲明**: 本系統僅供學習和研究用途。交易涉及風險，請自行承擔責任。
