# CPB Trading V6 Model Training Guide (Colab)

完整的 V6 多幣種多時間框架模型訓練指南

## 概述

本指南將幫助你在 Google Colab 上訓練 V6 版本的模型，支援：

- **20+ 個主要加密貨幣**
- **3 個時間框架**：1d（日線）、1h（小時線）、15m（15分鐘線）
- **多步序列預測**：30 根 K 線輸入 → 10 根 K 線輸出
- **OHLC 完整預測**：Open, High, Low, Close
- **自動上傳到 HuggingFace**：models_v6 目錄

## 準備工作

### 1. 準備 HuggingFace 權限

1. 登入 [HuggingFace](https://huggingface.co)
2. 進入設定 → Access Tokens
3. 建立新的 Token（任何權限即可）
4. 複製 Token

### 2. 在 Colab 中設置 HF Token

1. 打開 Google Colab
2. 在左側菜單 "Secrets" 中新增密鑰
3. Key: `HF_TOKEN`
4. Value: 粘貼你的 HF Token

## Colab 訓練步驟

### 第一步：克隆 Repo

```bash
!git clone https://github.com/caizongxun/cpb-trading-web.git
%cd cpb-trading-web
```

### 第二步：安裝依賴

```bash
!pip install -q yfinance pandas numpy tensorflow scikit-learn huggingface-hub
```

### 第三步：啟動訓練

```bash
!python train_v6_models_multiframe.py
```

## 訓練配置

在 `train_v6_models_multiframe.py` 中可自訂以下參數：

### 支援的幣種（COINS）

```python
COINS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BNB': 'BNB-USD',
    'SOL': 'SOL-USD',
    'XRP': 'XRP-USD',
    'ADA': 'ADA-USD',
    'DOGE': 'DOGE-USD',
    'AVAX': 'AVAX-USD',
    'LINK': 'LINK-USD',
    'DOT': 'DOT-USD',
    'LTC': 'LTC-USD',
    'ATOM': 'ATOM-USD',
    'UNI': 'UNI-USD',
    'MATIC': 'MATIC-USD',
    'NEAR': 'NEAR-USD',
    'FTM': 'FTM-USD',
    'CRO': 'CRO-USD',
    'VET': 'VET-USD',
    'ICP': 'ICP-USD',
    'HBAR': 'HBAR-USD',
}
```

若要新增更多幣種，只需在字典中新增即可。

### 時間框架（TIMEFRAMES）

```python
TIMEFRAMES = {
    '1d': {'period': '2y', 'interval': '1d'},    # 2 years of daily data
    '1h': {'period': '60d', 'interval': '1h'},   # 60 days of hourly data
    '15m': {'period': '14d', 'interval': '15m'}, # 14 days of 15-min data
}
```

### 模型參數（MODEL_PARAMS）

```python
MODEL_PARAMS = {
    'lookback': 30,              # Input sequence length
    'forecast': 10,              # Output sequence length
    'lstm_units': 128,           # LSTM layer units
    'dropout': 0.2,              # Dropout rate
    'dense_units': 64,           # Dense layer units
    'epochs': 100,               # Maximum training epochs
    'batch_size': 32,            # Batch size
    'validation_split': 0.2,     # Train/val split
    'early_stopping_patience': 15, # Early stopping patience
}
```

## 輸出結構

訓練完成後，模型會上傳到 HF 的 `models_v6` 目錄：

```
zongowo111/cpb-models
├── models_v6/
│   ├── BTC_1d.h5                    # 比特幣日線模型
│   ├── BTC_1d_metrics.json          # 模型評估指標
│   ├── ETH_1h.h5                    # 以太坊小時線模型
│   ├── ETH_1h_metrics.json
│   ├── SOL_15m.h5                   # Solana 15分鐘線模型
│   ├── SOL_15m_metrics.json
│   └── ... (所有幣種和時間框架組合)
```

## 模型評估指標

每個模型都會生成包含以下指標的 JSON 文件：

```json
{
  "Open": {
    "MAE": 123.45,
    "RMSE": 234.56,
    "MAPE": 2.34
  },
  "High": {
    "MAE": 234.56,
    "RMSE": 345.67,
    "MAPE": 3.45
  },
  "Low": {
    "MAE": 123.45,
    "RMSE": 234.56,
    "MAPE": 2.34
  },
  "Close": {
    "MAE": 234.56,
    "RMSE": 345.67,
    "MAPE": 3.45
  }
}
```

其中：
- **MAE** (Mean Absolute Error)：平均絕對誤差
- **RMSE** (Root Mean Squared Error)：根均方誤差
- **MAPE** (Mean Absolute Percentage Error)：平均絕對百分比誤差（%）

## 模型架構

### 輸入層
- Shape: (lookback=30, features=4) → OHLC 序列
- 每個序列代表過去 30 根 K 線的 OHLC 數據

### 編碼器層
- LSTM 層：128 units，activation='relu'，return_sequences=True
- Dropout：0.2

### 解碼器層
- LSTM 層：128 units，activation='relu'，return_sequences=False
- Dropout：0.2
- Dense 層：64 units，activation='relu'

### 輸出層
- Dense 層：forecast * n_features = 10 * 4 = 40 units
- 重塑為 (forecast=10, features=4) → 未來 10 根 K 線的 OHLC

## 訓練時間估計

在 Colab GPU 上：

| 幣種數 | 時間框架數 | 估計時間 |
|--------|-----------|--------|
| 20     | 3         | 8-12 小時 |
| 5      | 3         | 2-3 小時  |
| 1      | 3         | 20-30 分鐘 |

## 故障排除

### 問題 1：HF Token 未加載

```
HF Token not found in Colab Secrets.
```

**解決方案**：
1. 在 Colab 左側菜單中新增密鑰
2. 或在代碼中手動設置：

```python
HF_TOKEN = "your_token_here"
```

### 問題 2：數據不足

```
INSUFFICIENT DATA FOR XXX (15m). SKIPPING.
```

**解決方案**：
- 15m 時間框架需要至少 14 天的數據
- 某些小幣種可能沒有足夠的歷史數據
- 跳過是正常的，程序會繼續訓練其他幣種

### 問題 3：超時或內存不足

**解決方案**：
- 減少訓練幣種數量
- 減少 `epochs` 或 `batch_size`
- 或分批訓練（例如先訓練 1d，再訓練 1h）

## 本地運行

若要在本地運行而不上傳到 HF：

```bash
python train_v6_models_multiframe.py
```

模型將保存到本地 `models_v6/` 目錄。

## 後續步驟

訓練完成後，可以：

1. **集成到推理 API**：
   - 下載模型並集成到 `app_v6.py`
   - 支援動態加載 HF 上的模型

2. **實時推理**：
   - 實現滑動窗口推理
   - 每新增一根 K 線就重新預測下一個 10 根

3. **模型評估**：
   - 在實時數據上測試模型
   - 調整模型參數並重新訓練

## 相關文件

- 訓練腳本：`train_v6_models_multiframe.py`
- HF 模型庫：https://huggingface.co/datasets/zongowo111/cpb-models
- 推理 API：`app_v6.py`（待開發）

## 快速命令

Colab 單一代碼塊：

```python
# 1. 克隆和安裝
!git clone https://github.com/caizongxun/cpb-trading-web.git
%cd cpb-trading-web
!pip install -q yfinance pandas numpy tensorflow scikit-learn huggingface-hub

# 2. 訓練
!python train_v6_models_multiframe.py
```

完成！所有模型將自動上傳到 HuggingFace。
