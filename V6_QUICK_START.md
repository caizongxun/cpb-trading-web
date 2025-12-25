# V6 快速開始 - 30秒上手

## Colab 一鍵訓練

在 Google Colab 中依次執行以下代碼塊：

### 步驟 1：準備環境

```python
# 克隆倉庫
!git clone https://github.com/caizongxun/cpb-trading-web.git
%cd cpb-trading-web

# 安裝依賴
!pip install -q yfinance pandas numpy tensorflow scikit-learn huggingface-hub
```

### 步驟 2：設置 HF 密鑰（在左菜單 "Secrets" 中）

- 密鑰名: `HF_TOKEN`
- 值: 你的 [HuggingFace API Token](https://huggingface.co/settings/tokens)

### 步驟 3：啟動訓練

```python
!python train_v6_models_multiframe.py
```

完成！所有模型會自動保存到:
```
https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v6
```

---

## 訓練統計

| 配置 | 說明 |
|------|------|
| 幣種 | 20 個主要加密貨幣 |
| 時間框架 | 1d (日線) / 1h (小時線) / 15m (15分鐘線) |
| 總組合 | 60 個模型 |
| 輸入 | 30 根 K 線 (OHLC) |
| 輸出 | 10 根 K 線預測 (OHLC) |
| GPU 時間 | ~8-12 小時（完整訓練） |

---

## 支持的幣種

```
BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT 
LTC ATOM UNI MATIC NEAR FTM CRO VET ICP HBAR
```

---

## 自定義訓練

### 減少訓練時間

編輯 `train_v6_models_multiframe.py`:

```python
# 僅訓練 BTC 和 ETH
COINS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
}

# 減少 epoch
MODEL_PARAMS = {
    'epochs': 30,  # 默認 100
    ...
}
```

### 修改時間框架

```python
TIMEFRAMES = {
    '1d': {'period': '2y', 'interval': '1d'},
    # '1h': {'period': '60d', 'interval': '1h'},  # 注釋掉，跳過小時線
    # '15m': {'period': '14d', 'interval': '15m'},
}
```

---

## 模型文件

訓練完成後，每個模型包含：

1. **模型文件** (`.h5`)
   - Keras/TensorFlow 格式
   - 可直接加載進行推理

2. **評估指標** (`.json`)
   ```json
   {
     "Close": {
       "MAE": 123.45,
       "RMSE": 234.56,
       "MAPE": 2.34
     },
     ...
   }
   ```

---

## HF 存儲結構

```
zongowo111/cpb-models (Dataset)
└── models_v6/
    ├── BTC_1d.h5
    ├── BTC_1d_metrics.json
    ├── BTC_1h.h5
    ├── BTC_1h_metrics.json
    ├── BTC_15m.h5
    ├── BTC_15m_metrics.json
    ├── ETH_1d.h5
    ├── ETH_1d_metrics.json
    └── ... (其他幣種和時間框架)
```

---

## 預期輸出

訓練完成時會顯示：

```
================================================================================
                        TRAINING SUMMARY
================================================================================

Success: 60/60
Failed: 0/60

Detailed Results:
  ✓ BTC_1d: SUCCESS
  ✓ BTC_1h: SUCCESS
  ✓ BTC_15m: SUCCESS
  ✓ ETH_1d: SUCCESS
  ...

================================================================================
Models saved to: models_v6/
HF Repository: https://huggingface.co/datasets/zongowo111/cpb-models
================================================================================
```

---

## 後續集成

完成訓練後，模型可以集成到：

1. **推理 API** (`app_v6.py`)
   - 實時加載 HF 模型
   - 支持動態預測

2. **Web 儀表板**
   - 顯示 10 根未來 K 線預測
   - 實時更新推理結果

3. **交易信號**
   - 基于 OHLC 預測的趨勢判斷
   - 風險管理邏輯

---

## 故障排查

### 連接超時

```
Connection refused / Timeout
```

**解決**: 重試或使用代理

### 內存不足

```
OOM / Runtime crashed
```

**解決**: 減少幣種或時間框架

### HF 上傳失敗

```
HF Upload FAILED
```

**解決**: 檢查 Token 權限，或改為本地保存

---

## 詳細指南

完整說明見 [COLAB_V6_TRAINING_GUIDE.md](COLAB_V6_TRAINING_GUIDE.md)

---

**Ready? 打開 Google Colab 開始訓練！**
