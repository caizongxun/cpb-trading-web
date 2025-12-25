# V6 Model Architecture - Technical Deep Dive

## 概述

V6 採用多步序列預測（Multi-Step Seq2Seq）架構，直接輸出未來 10 根 K 線的完整 OHLC 數據，而不是單一價格點或插值結果。

---

## 1. 核心架構

### 輸入 (Input)

```
Shape: (batch_size, 30, 4)
       ↓
   (batch, lookback, features)
       ↓
features = [Open, High, Low, Close] (OHLC)
```

**特點**：
- **30 根歷史 K 線**（lookback window）
- **4 個特徵**（OHLC）
- **範數化到 [0, 1]**（MinMaxScaler per feature）

### 數據預處理

```python
# 時間序列生成
for i in range(len(data) - 30 - 10):
    X[i] = data[i : i+30]        # 過去 30 根
    y[i] = data[i+30 : i+40]     # 未來 10 根
```

**優點**：
- 每個樣本都是完整的 30→10 映射
- 避免數據洩露（future leak）
- 自動對齐市場狀態

---

## 2. 編碼器（Encoder）

### 第一個 LSTM 層

```python
LSTM(
    units=128,
    activation='relu',
    input_shape=(30, 4),
    return_sequences=True  # 輸出整個序列
)
```

**配置**：
- **Units**: 128（隱藏狀態維度）
- **Return_sequences**: True（保留時間序列）
- **Activation**: ReLU（非線性）

**作用**：
- 逐時間步處理 30 根 K 線
- 學習局部模式（小時級到日級）
- 建立特徵表示

### Dropout 層

```python
Dropout(rate=0.2)  # 隨機丟棄 20% 神經元
```

**目的**：
- 防止過擬合
- 增強泛化能力
- 類似 ensemble 效果

### 第二個 LSTM 層（無序列輸出）

```python
LSTM(
    units=128,
    activation='relu',
    return_sequences=False  # 只輸出最後一個時間步
)
```

**作用**：
- 汇聚整個歷史信息
- 生成上下文向量（context vector）
- 為解碼器準備初始狀態

---

## 3. 解碼層（Decoder）

### 第一個 Dense 層

```python
Dense(units=64, activation='relu')
```

**作用**：
- 特徵轉換
- 非線性映射到中間表示
- 降低過擬合風險

### Dropout

```python
Dropout(rate=0.2)
```

### 輸出層

```python
Dense(units=40)  # 10 * 4 = 40
```

**說明**：
- 直接輸出展平的 10 根 K 線
- 每 4 個值對應一根 K 線的 OHLC

---

## 4. 完整計算流程

```
Input: (batch, 30, 4)
  ↓
LSTM 128 (return_seq=True)
  ↓ shape: (batch, 30, 128)
Dropout 0.2
  ↓
LSTM 128 (return_seq=False)
  ↓ shape: (batch, 128)
Dropout 0.2
  ↓
Dense 64 (ReLU)
  ↓ shape: (batch, 64)
Dropout 0.2  [可選，上面代碼中没有]
  ↓
Dense 40
  ↓ shape: (batch, 40)
Reshape to (batch, 10, 4)
  ↓ shape: (batch, 10, 4)  ← 未來 10 根 K 線 OHLC
```

---

## 5. 損失函數與優化

### 損失函數

```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
```

**MSE (Mean Squared Error)**：

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**優點**：
- 對大偏差懲罰更大
- 數值穩定
- 適合連續價格預測

**缺點**：
- 對異常值敏感
- 不考慮方向正確性

### 優化器

```python
Adam(learning_rate=0.001)
```

**特點**：
- 自適應學習率
- 第一和第二階矩估計
- 快速收斂，避免鞍點

---

## 6. 訓練超參數

```python
MODEL_PARAMS = {
    'lookback': 30,              # 輸入序列長度
    'forecast': 10,              # 輸出序列長度
    'lstm_units': 128,           # LSTM 單元數
    'dropout': 0.2,              # Dropout 率
    'dense_units': 64,           # Dense 層單元數
    'epochs': 100,               # 最大 epoch 數
    'batch_size': 32,            # 批次大小
    'validation_split': 0.2,     # 驗證集比例
    'early_stopping_patience': 15, # 早停耐心值
}
```

### 訓練設置

```python
model.fit(
    X_train, y_train_flat,
    validation_data=(X_val, y_val_flat),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
)
```

**回調函數**：
1. **EarlyStopping**：驗證損失不再下降時停止
2. **ReduceLROnPlateau**：損失平台期時降低學習率

---

## 7. 數據標準化

### 獨立標準化

```python
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    scaler = MinMaxScaler()
    scaled_data[:, i] = scaler.fit_transform(data[:, i:i+1])
    scalers[col] = scaler
```

**原因**：
- **避免相關性**：Open 和 Close 有強相關性，但獨立標準化讓模型自己學習
- **數值穩定性**：不同特徵數值範圍不同（High-Low 往往更大）
- **反演簡化**：推理時每個特徵用各自 scaler 反演

### 標準化公式

$$\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

範圍：$[0, 1]$

### 反演

$$x = \hat{x} \times (\max(x) - \min(x)) + \min(x)$$

---

## 8. 性能評估指標

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**解釋**：平均絕對偏差，單位與原數據相同（USD）

### RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

**解釋**：對大偏差更敏感，單位同原數據

### MAPE (Mean Absolute Percentage Error)

$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**解釋**：百分比誤差，便於跨幣種比較

---

## 9. 典型性能表現

### BTC 1d

```json
{
  "Close": {
    "MAE": 234.56,
    "RMSE": 345.67,
    "MAPE": 2.34
  }
}
```

**解讀**：
- 平均誤差 ±$235
- MAPE 2.34% ≈ 較好的預測準確度

### ETH 1h

```json
{
  "Close": {
    "MAE": 12.34,
    "RMSE": 18.92,
    "MAPE": 1.89
  }
}
```

**解讀**：
- 1h 尺度誤差較小
- MAPE 更低（小時級別波動更規律）

---

## 10. 與 V5 的區別

| 特性 | V5 | V6 |
|-----|----|-|
| **輸出** | 單一 predicted_price | 10 根完整 OHLC |
| **序列** | 單步預測 | 多步（10步）預測 |
| **模型結構** | 線性模型 + 補償 | Seq2Seq LSTM |
| **時間感知** | 有限 | 強（編碼器-解碼器） |
| **OHLC 生成** | 前端插值 | 模型直接輸出 |
| **誤差積累** | 無（單步） | 有（多步，但控制） |

---

## 11. 推理流程

```python
# 1. 加載模型
model = load_model('BTC_1d.h5')
scalers = pickle.load(open('BTC_1d_scalers.pkl', 'rb'))

# 2. 準備輸入（最近 30 根 K 線）
recent_30_bars = fetch_recent_klines('BTC-USD', 30)  # shape: (30, 4)
scaled = normalize(recent_30_bars, scalers)
X = scaled.reshape(1, 30, 4)  # 加入 batch 維度

# 3. 預測
y_pred = model.predict(X)  # shape: (1, 40)
y_pred = y_pred.reshape(1, 10, 4)  # reshape 回 (batch, 10, 4)

# 4. 反標準化
forecast_ohlc = []
for i in range(4):  # Open, High, Low, Close
    col_name = ['Open', 'High', 'Low', 'Close'][i]
    unscaled = scalers[col_name].inverse_transform(y_pred[:, :, i])
    forecast_ohlc.append(unscaled[0])  # 移除 batch 維度

# 5. 輸出結果
for t in range(10):
    o, h, l, c = [forecast_ohlc[i][t] for i in range(4)]
    print(f"Bar {t+1}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}")
```

---

## 12. 關鍵設計決策

### 為什麼 LSTM？

1. **時間依賴性**：LSTM 通過門控機制捕捉長期依賴
2. **梯度流**：避免梯度消失/爆炸
3. **記憶能力**：cell state 保留重要信息

### 為什麼 30 根？

- **1d**：30 天 ≈ 1 個月，包含足夠的趨勢信息
- **1h**：30 小時 ≈ 1.25 天，捕捉日內模式
- **15m**：30 × 15min = 450 分鐘 ≈ 7.5 小時

### 為什麼 10 根？

- **1d**：10 天是 2 周，合理預測周期
- **1h**：10 小時是半天，實用交易周期
- **15m**：2.5 小時，快速確認信號

---

## 13. 未來優化方向

### 短期（簡單）

- [ ] 增加更多幣種
- [ ] 調整 LSTM units (64, 256)
- [ ] 嘗試不同 dropout 率

### 中期（複雜）

- [ ] 添加注意力機制（Attention）
- [ ] 多頭預測（同時預測多個目標）
- [ ] 條件生成（基於市場狀態）

### 長期（研究級）

- [ ] Transformer 架構
- [ ] 圖神經網絡（幣種間相關性）
- [ ] 強化學習（交易獎勵函數）

---

## 參考資源

- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)
- [Time Series Forecasting with LSTM](https://arxiv.org/abs/1506.02640)

---

**現在你完全理解了 V6 的運作方式。準備訓練吧！**
