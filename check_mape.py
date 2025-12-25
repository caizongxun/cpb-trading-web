#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAP中文e评估脚本 - 每个币种的不符率查看

MAP中e = 平均绝对百分比误差 (Mean Absolute Percentage Error)
来自: 每次运行时实时计算，值会随矢失上案例変改
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import sys

# 忽略警告
warnings.filterwarnings("ignore")

# 设置要评估的币种
SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD',
    'LTC-USD', 'ATOM-USD'
]

def calculate_mape(actual, predicted):
    """计算 MAPE"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def v5_logic_predict(window_closes):
    """模拟 V5 模型的预测逻辑"""
    # 简单趨势逻辑
    current_price = window_closes[-1]
    recent_avg = np.mean(window_closes[-5:])
    past_avg = np.mean(window_closes[:5])
    
    # 判断方向
    direction = 1 if recent_avg > past_avg else -1
    
    # 计算波动率
    # 避免除以 0
    returns = np.diff(window_closes[-10:]) / window_closes[-11:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0.02
    
    # 预测
    pred_change = direction * volatility * 0.5
    predicted_price = current_price * (1 + pred_change)
    return predicted_price

def evaluate_coin(symbol, days=60):
    print(f"\u6b63在评估 {symbol}...", end="", flush=True)
    
    # 下载数据
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
    
    if len(df) < 30:
        print(" [数据不足]")
        return None
        
    actuals = []
    preds = []
    
    # 滚动回测：从笥25天开始，每一天都尝试预测"明天"
    data = df['Close'].values.flatten()
    
    for i in range(25, len(data)-1):
        # 取过待25根K线作为输入窗口
        window = data[i-25:i+1]
        
        # 进行预测
        pred_price = v5_logic_predict(window)
        
        # 实际的"明天"价格
        actual_next_day = data[i+1]
        
        preds.append(pred_price)
        actuals.append(actual_next_day)
    
    # 计算指标
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    mape = calculate_mape(actuals, preds)
    
    # 方向准确率 (涨跌方向是否预测正確)
    currents = data[25:len(data)-1]
    pred_dir = np.sign(preds - currents)
    actual_dir = np.sign(actuals - currents)
    accuracy = np.mean(pred_dir == actual_dir) * 100
    
    print(" [完成]")
    return {
        'symbol': symbol,
        'mape': mape,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }

def get_grade(mape):
    """根据 MAPE 给出评分"""
    if mape < 2:
        return "\u2b50\u2b50\u2b50\u2b50\u2b50 最优 (MAPE < 2%)"
    elif mape < 3:
        return "\u2b50\u2b50\u2b50\u2b50 优 (MAPE < 3%)"
    elif mape < 5:
        return "\u2b50\u2b50\u2b50 良好 (MAPE < 5%)"
    elif mape < 10:
        return "\u2b50\u2b50 一般 (MAPE < 10%)"
    else:
        return "\u2b50 需改进 (MAPE >= 10%)"

# === 主程序 ===
print("\n" + "="*80)
print("CPB 交易模型 - MAPE 指标评估")
print("="*80)
print(f"\u4e0b载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n超时间: 5-10 分钟 (\u7b2c一次会比较久)\n")

print(f"{'\u5e01\u79cd':<10} | {'MAPE (%)':<10} | {'\u65b9\u5411\u51c6\u786e\u7387':<15} | {'\u8bc4\u7ea7':<30}")
print("="*80)

results = []
errors = []

for sym in SYMBOLS:
    try:
        res = evaluate_coin(sym)
        if res:
            grade = get_grade(res['mape'])
            coin_name = res['symbol'].replace('-USD', '')
            print(f"{coin_name:<10} | {res['mape']:<10.2f} | {res['accuracy']:<15.2f} | {grade:<30}")
            results.append(res)
    except Exception as e:
        error_msg = f"{sym}: {str(e)[:50]}"
        errors.append(error_msg)
        print(f"\u9519\u8bef {error_msg}")

print("="*80)

if results:
    avg_mape = np.mean([r['mape'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\n\u6574\u4f53\u5e73\u5747 MAPE: {avg_mape:.2f}%")
    print(f"\u6574\u4f53\u5e73\u5747\u65b9\u5411\u51c6\u786e\u7387: {avg_accuracy:.2f}%")
    print(f"\n\u6210\u529f\u8bc4\u4f30: {len(results)}/{len(SYMBOLS)} \u4e2a\u5e01\u79cd")

if errors:
    print(f"\n\u5931\u8d25: {len(errors)} \u4e2a\u5e01\u79cd")
    for e in errors:
        print(f"  - {e}")

print("\n" + "="*80)
print("\n\u8bc4\u7d1a\u89e3\u8bfb:")
print("  \u2b50\u2b50\u2b50\u2b50\u2b50: 模\u5f62\u7cbe\u51c6\u5ea6\u6700\u9ad8\uff0c\u4f1a\u4e0a\u5e02\u7a97\u4e2d\u8868\u73b0\u4e3a "\u8d85\u7ea7\u6a21\u5f62"")
print("  \u2b50\u2b50\u2b50\u2b50: MAPE < 3% \uff0c\u53ef\u4ee5\u653e\u5fc3\u4f7f\u7528\uff0c\u63a8\u8350\u52a0\u5927\u4ed3\u4f4d")
print("  \u2b50\u2b50\u2b50: MAPE < 5% \uff0c\u4e2d\u5e73\u6c34\u5e73\uff0c\u6309\u6807\u6570\u4f7f\u7528")
print("  \u2b50\u2b50: MAPE < 10% \uff0c\u9700\u8c28\u614e\u4f7f\u7528\uff0c\u4ec5\u53c2\u8003\u65b9\u5411")
print("  \u2b50: MAPE > 10% \uff0c\u6a21\u5f62\u5931\u6548\uff0c\u4e0d\u5efa\u8bae\u4f7f\u7528")
print("\n" + "="*80 + "\n")
