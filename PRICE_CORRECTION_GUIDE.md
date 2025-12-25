# 价格修正指南 - MAPE 偏移自动校正

## 问题

票前的预测价格经常偏离很远，壹些情况下会偏高或偏低。

## 原因

**MAPE (Mean Absolute Percentage Error) 传闘**:

1. **模型输出不是价格**→ 是价格変化百分比
2. **模型性能偏差**→ 某些方向的预测试错
3. **正常化/反正常化的累积误差**

## 解决方案

### 1. 低改改 V5_APP_WITH_MARKET_ANALYSIS.py

新文件名: `V5_APP_WITH_MARKET_ANALYSIS.py`

GitHub 地址: 
```
https://raw.githubusercontent.com/caizongxun/cpb-trading-web/main/V5_APP_WITH_MARKET_ANALYSIS.py
```

### 2. 最主要特性: PriceCorrector 类

```python
class PriceCorrector:
    """处理模型预测的价格偏移自动校正"""
    
    def correct_predicted_price(
        self,
        current_price: float,
        predicted_price: float,
        historical_prices: List[float],
        confidence: float = 0.65
    ) -> Dict:
        """
        自动校正预测价格，裦腐:
        
        1. 计算歴史上涨下跌幅度 (标准差)
        2. 检查预测是否超过合理范围
        3. 三倍规则: 如果预测超过3倍歴史波动，接下校正
        4. 应用边界: 确保价格不会走极端
        """
```

## 修正機制

### 步骤6: 历史波动分析

```
计算最近20根K线的收盤正常化收盤率
     ↓
计算标准差 → 这是历史上涨下跌的典上涨下跌幅度
```

### 步骤7: 检查预测是否合理

```
如果 |predicted_pct_change| > historical_volatility × 3:
    预测偏离太大 → 需要校正
否则:
    预测合理 → 不需要校正
```

### 步骤8: 应用校正根数

```
预测偏高 (正水掤):
    corrected_price = current × (1 + pct_change / 1.15)
    → 降低预测幅度

预测偏低 (負水掤):
    corrected_price = current × (1 + pct_change / 0.85)
    → 提高预测幅度
```

### 步骤9: 应用边界

```
max_allowed_change = historical_volatility × 5

max_price = current × (1 + max_allowed_change)
min_price = current × (1 - max_allowed_change)

corrected_price = clamp(corrected_price, min_price, max_price)
→ 确保价格不会走极端
```

## 使用方法

### 步骤1: 敲换文件

取代你的旧 `app.py`:

```bash
# 备份旧文件 (可选)
python app.py  # 改名为 app_backup.py

# 下載新文件
wget https://raw.githubusercontent.com/caizongxun/cpb-trading-web/main/V5_APP_WITH_MARKET_ANALYSIS.py -O app.py

# 或眉敳拶
栽水V5_APP_WITH_MARKET_ANALYSIS.py 的下允根据 app.py
```

### 步骤2: 重新啟动 API

```bash
python app.py
```

应該看到:

```
================================================================================
               CPB Trading Web - V5 Model (HYBRID VERSION)
================================================================================

Model Version: V5 (HYBRID)
...
⚠  PRICE CORRECTION ENABLED!
   - Automatic MAPE offset correction
   - Historical volatility-based bounds
   - Smart confidence-weighted adjustments

================================================================================
```

### 步骤3: 測試

前端測試（一旦区动）:

```
1. 選擇幣種: BTC
2. 選擇時間框架: 1d
3. 點擊「獲取預測」
4. 粗鎨預測价格是否比之前更接近當前价格
```

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

看 predicted_price 是否合理：

```json
{
  "current_price": 42500.00,
  "predicted_price": 42650.50,      ⬅ 校正後的价格 (合理范围内)
  "log_return": 0.0035,
  "volatility": {
    "current": 0.45,
    "predicted": 0.35                ⬅ 子动了
  },
  "recommendation": "BUY"
}
```

## 壹些需試门梏

### 问题 1: 价格仍奪偏离很远

**栙本**:

1. 检查 `historical_prices` 是否正確 (■■■20根)
2. 查看 `volatility_predicted` - 如果太高，可粗改改 correctorl 细吪数
3. 确保 `correction_factors` 合理

### 问题 2: 万丈一不有推荐信号

**原因**: 可能是模型区动预测正常→ 认小预测价格 (HOLD)

**修复**: 调整 `confidence` 阈值或下閟 `volatility` 报警阈值

### 问题 3: 不窗是歴史波动计算

**方案**: 跟改去傲去解算回歴史波动緣捡:

```python
# 原始的
 hist_volatility = np.std(hist_returns)  # 标准差

# 改为子平均唐正常化歴史收盤率
hist_volatility = np.mean(np.abs(hist_returns))  # 绝对值平均
```

## 縪床指標

API 回應中的 `correction_info` 欄位:

```json
{
  "original_predicted": 42800.00,      ⬅ 本沾的预测价格
  "corrected_predicted": 42650.50,     ⬅ 校正後的价格
  "correction_applied": true,          ⬅ 是否應用校正
  "correction_pct": -0.35,             ⬅ 校正百分比 (-0.35% = 降低了)
  "hist_volatility": 0.018,            ⬅ 歴史波动王
  "max_allowed_change": 0.09           ⬅ 最大允许变化 (9%)
}
```

## 下一步改进

1. **物模形所且模型載入** - 將 `demo_mode=True` 改为实際使用生事誁模型
2. **动态调整校正因子** - 你洠敵由推社改風主詷鼠了
3. **基於市场雜駕的上撤上晷** - 依賬人寸回更新 `correction_factors`

## 需要更多幫助?

查看 `V5_APP_WITH_MARKET_ANALYSIS.py` 頁併的決况詳細樣例。
