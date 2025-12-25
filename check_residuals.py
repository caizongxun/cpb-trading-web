#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual (残差) 计算脚本
计算每个币种的系统性偏差 (Systematic Bias)

残差 = 实际价 - 预测价
- 残差 > 0: 模型整体低估
- 残差 < 0: 模型整体高估
- 残差 ~ 0: 模型无偏差
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
    print(f"正在计算 {symbol}...", end="", flush=True)
    
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
        
        # 计算统计指标
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        min_residual = np.min(residuals)
        max_residual = np.max(residuals)
        median_residual = np.median(residuals)
        
        # 系统偏差不平衡度 (是否一致残差)
        # 如果 std 很小，说明残差很稳定、粗整（便于校正）
        # 如果 std 很大，说明残差不稳定、随机（需要改进模型）
        bias_stability = "稳定" if std_residual < abs(mean_residual) else "不稳定"
        
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
        interpretation = "模型不需调整，非常准"
    elif abs(mean_residual) < 500:
        interpretation = "小幅偏差，低估/高估很齐"
    elif abs(mean_residual) < 1000:
        interpretation = "中等偏差，建议校正"
    else:
        interpretation = "大的系统性偏差，需要校正"
    
    bias_quality = "稳定偏差（容易修复）" if std_residual < abs(mean_residual) else "不稳定偏差（难改善）"
    
    return f"{interpretation} | {bias_quality}"

# === 主程序 ===
print("\n" + "="*120)
print("残差 (Residual) 分析 - 计算每个币种的系统性偏差")
print("="*120)
print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"每个币种会回测最近 90 天的数据，正常一次配置需要 5-15 分钟\n")

results = []
errors = []

for sym in SYMBOLS:
    res = calculate_residuals(sym, days=90)
    if res:
        results.append(res)
    else:
        errors.append(sym)

print("\n" + "="*120)

# 打印表头
header_symbol = "币种"
header_mean = "平均残差"
header_std = "标准差"
header_median = "中位数"
header_range = "最小~最大"
header_interp = "为什么"

print(f"{header_symbol:<10} | {header_mean:<15} | {header_std:<15} | {header_median:<15} | {header_range:<30} | {header_interp:<40}")
print("="*120)

for res in sorted(results, key=lambda x: abs(x['mean_residual'])):
    symbol_name = res['symbol'].replace('-USD', '')
    mean_res = res['mean_residual']
    std_res = res['std_residual']
    median_res = res['median_residual']
    min_res = res['min_residual']
    max_res = res['max_residual']
    interpretation = interpret_residual(mean_res, std_res)
    
    # 增加符号表示低估/高估
    sign = "↓" if mean_res > 0 else "↑"
    
    range_str = f"{min_res:.0f}~{max_res:.0f}$"
    print(f"{symbol_name:<10} | {mean_res:>14.2f}$ | {std_res:>14.2f}$ | {median_res:>14.2f}$ | {range_str:<30} | {interpretation:<40}")

print("="*120)

# 统计汇总
print(f"\n统计汇总:")
if results:
    avg_mean = np.mean([r['mean_residual'] for r in results])
    avg_median = np.median([r['median_residual'] for r in results])
    avg_std = np.mean([r['std_residual'] for r in results])
    
    print(f"  总体平均残差: {avg_mean:.2f}$")
    print(f"  总体中位数: {avg_median:.2f}$")
    print(f"  总体标准差: {avg_std:.2f}$")
    
    if avg_mean > 0:
        print(f"  推估模型: 整体低估 (↓ 所有币种平均偏低 {avg_mean:.2f}$)")
    elif avg_mean < 0:
        print(f"  推估模型: 整体高估 (↑ 所有币种平均偏高 {abs(avg_mean):.2f}$)")
    else:
        print(f"  推估模型: 完全无偏差")
    
    print(f"  成功评估: {len(results)}/{len(SYMBOLS)} 个币种")

if errors:
    print(f"  失败: {len(errors)} 个币种")

print("\n" + "="*120)
print("\n残差解读指南:")
print("""
  残差 = 实际价 - 预测价
  
  残差 > 0 (正值):    模型整体低估了
  残差 < 0 (负值):    模型整体高估了
  残差 ~ 0:           模型无偏差，异常准确
  
  标准差 (残差的波动范围):  
    - 低 (低于平均值):   残差很稳定 -> 容易校正 (上限 + 残差值)
    - 高 (高于平均值):   残差非常不稳定 -> 想要校正效果草
  
  关键信号: 残差趋势图
    - 如果残差关于 0 查看 足是 正值或负值，
    - 总一个方向摇摇欲坠 -> 可以稳定校正
""".strip())

print("\n" + "="*120 + "\n")
