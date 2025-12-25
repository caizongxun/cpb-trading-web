#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual (残差) 计算脚本
计算每个币种的系统性偏差 (Systematic Bias)

残差 = 实际价 - 预测价
- 残差 > 0: 模型整体低估
- 残差 < 0: 模型整体高估
- 残差 ∼ 0: 模型无偏差
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# 设置要评估的币种
SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD',
    'LTC-USD', 'ATOM-USD'
]

def v5_logic_predict(window_closes):
    """模拟 V5 模型的预测逻辑"""
    current_price = window_closes[-1]
    recent_avg = np.mean(window_closes[-5:])
    past_avg = np.mean(window_closes[:5])
    
    direction = 1 if recent_avg > past_avg else -1
    
    returns = np.diff(window_closes[-10:]) / window_closes[-11:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0.02
    
    pred_change = direction * volatility * 0.5
    predicted_price = current_price * (1 + pred_change)
    return predicted_price

def calculate_residuals(symbol, days=90):
    """
    计算一个币种的残差序列
    残差 = 实际价 - 预测价
    """
    print(f"\u6b63在计算 {symbol}...", end="", flush=True)
    
    try:
        # 下载数据
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        
        if len(df) < 30:
            print(" [数据不足]")
            return None
        
        residuals = []
        dates = []
        actuals = []
        preds = []
        
        # 滚动回测
        data = df['Close'].values.flatten()
        timestamps = df.index
        
        for i in range(25, len(data)-1):
            window = data[i-25:i+1]
            
            # V5 预测
            pred_price = v5_logic_predict(window)
            
            # 实际的明天价格
            actual_next_day = data[i+1]
            
            # 计算残差
            residual = actual_next_day - pred_price
            
            residuals.append(residual)
            actuals.append(actual_next_day)
            preds.append(pred_price)
            dates.append(timestamps[i+1])
        
        residuals = np.array(residuals)
        
        # 计算筱计指标
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        min_residual = np.min(residuals)
        max_residual = np.max(residuals)
        median_residual = np.median(residuals)
        
        # 系统偏差不平衡度 (是否一致残差)
        # 如果 std 很小，说明残差很稳定、粗整（便于校正）
        # 如果 std 很大，说明残差不稳定、随机（需要改进模形）
        bias_stability = "\u7a33\u5b9a" if std_residual < abs(mean_residual) else "\u4e0d\u7a33\u5b9a"
        
        print(" [完成]")
        
        return {
            'symbol': symbol,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'median_residual': median_residual,
            'min_residual': min_residual,
            'max_residual': max_residual,
            'bias_stability': bias_stability,
            'sample_count': len(residuals),
            'residuals': residuals,
            'dates': dates,
            'actuals': actuals,
            'predictions': preds
        }
    
    except Exception as e:
        print(f" [错误: {str(e)[:30]}]")
        return None

def interpret_residual(mean_residual, std_residual):
    """
    解读残差的含义
    """
    if abs(mean_residual) < 100:  # 低于 100 USD
        interpretation = "模形不需调整，非常沺"
    elif abs(mean_residual) < 500:
        interpretation = "小幅偏差，低估/高估很轳"
    elif abs(mean_residual) < 1000:
        interpretation = "中等偏差，建议校正"
    else:
        interpretation = "大的系统性偏差，需要校正"
    
    bias_quality = "稳\u5b9a\u504f\u5dee\uff08\u5bb9\u6613\u4fee\u590d\uff09" if std_residual < abs(mean_residual) else "不\u7a33\u5b9a\u504f\u5dee\uff08\u96be\u6539\善\uff09"
    
    return f"{interpretation} | {bias_quality}"

# === 主程序 ===
print("\n" + "="*100)
print("\u6b8b差 (Residual) 分析 - 计算每个币种的系统性偏差")
print("="*100)
print(f"\u6267行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\u6bcf个币种会回测最近 90 \u5929\u7684数\u636e\uff0c正\u5e38\u4e00\u6b21\u914d\u7f6e\u9700\u8981 5-15 \u5206\u949f\n\n")

results = []
errors = []

for sym in SYMBOLS:
    res = calculate_residuals(sym, days=90)
    if res:
        results.append(res)
    else:
        errors.append(sym)

print("\n" + "="*100)
print(f"{'\u5e01\u79cd':<10} | {'\u5e73\u5747\u6b8b\u5dee':<15} | {'\u6807\u51c6\u5dee':<15} | {'\u4e2d\u4f4d\u6570':<15} | {'\u6700\u5c0f~\u6700\u5927':<30} | {'\u70ba\u59426':<30}")
print("="*100)

for res in sorted(results, key=lambda x: abs(x['mean_residual'])):
    symbol_name = res['symbol'].replace('-USD', '')
    mean_res = res['mean_residual']
    std_res = res['std_residual']
    median_res = res['median_residual']
    min_res = res['min_residual']
    max_res = res['max_residual']
    interpretation = interpret_residual(mean_res, std_res)
    
    # 增加符号表示低估/高估
    sign = "" if mean_res > 0 else ""
    
    print(f"{symbol_name:<10} | {mean_res:>14.2f}$ | {std_res:>14.2f}$ | {median_res:>14.2f}$ | {min_res:>10.2f}~{max_res:<10.2f}$ | {interpretation:<30}")

print("="*100)

# 统计汇总
(print(f"\n\u7edf\u8ba1汇\u603b:")
if results:
    print(f"  总体\u5e73\u5747\u6b8b\u5dee: {np.mean([r['mean_residual'] for r in results]):.2f}$")
    print(f"  总\u4f53\u4e2d\u4f4d\u6570: {np.median([r['median_residual'] for r in results]):.2f}$")
    print(f"  \u603b\u4f53\u6807\u51c6\u5dee: {np.mean([r['std_residual'] for r in results]):.2f}$")
    print(f"  \u63a8\u4f30\u6a21\u578b: {'\u6574\u4f53\u6574\u4f53\u4f4e\u4f30' if np.mean([r['mean_residual'] for r in results]) > 0 else '\u6574\u4f53\u6574\u4f53\u9ad8\u4f30'}")

if errors:
    print(f"  \u5e73\u5931\u8d25: {len(errors)} \u4e2a\u5e01\u79cd")

print("\n" + "="*100)
print("\n\u6b8b\u5dee\u89e3\u8bfb\u6307\u5357:")
print("""
  残\u5dee = \u5b9e\u9645\u4ef7 - \u9884\u6d4b\u4ef7
  
  \u6b8b\u5dee > 0 (\u6b63\u503c):    模\u5f62\u6574\u4f53\u4f4e\u4f30\u4e86\n  \u6b8b\u5dee < 0 (\u8ca0\u503c):    模\u5f62\u6574\u4f53\u9ad8\u4f30\u4e86
  残\u5dee \u223c 0:           \u6a21\u5f62\u65e0\u504f\u5dee\uff0c\u5f02\u5e38\u51c6\u786e
  
  \u6807\u51c6\u5dee (\u6b8b\u5dee\u7684\u6ce2\u52a8\u8303\u56f4):  
    - \u4f4e (\u4f4e\u4e8e\u5e73\u5747\u503c):   \u6b8b\u5dee\u5f88\u7a33\u5b9a -> \u5bb9\u6613\u6821\u6b63 (\u4e0a\u9694 + 1000)
    - \u9ad8 (\u9ad8\u4e8e\u5e73\u5747\u503c):   \u6b8b\u5dee\u975e\u5e38\u4e0d\u7a33\u5b9a -> \u60f3\u8981\u6821\u6b63\u6548\u679c\u8349\n  
  \u5173\u952e\u4fe1\u53f7: \u6b8b\u5dee \u8d8b\u52bf\u56fe
    - \u5982\u679c\u6b8b\u5dee\u5173\u4e8e \u67e5\u770b \u8db3\u662f \u6b63\u503c\u6216\u8ca0\u503c\uff0c
    - \u603b\u4e00\u4e2a\u65b9\u5411\u504a\u504a\u7eef -> \u53ef\u4ee5\u7a33\u5b9a\u6821\u6b63
""".strip())

print("\n" + "="*100 + "\n")
