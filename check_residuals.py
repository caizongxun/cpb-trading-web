#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual (残差) 计算脚本
计算每个币种的系统性偏差 (Systematic Bias)
高精度 (7位小数) - 对小币种有效

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

# 精度配置
PRECISION = 7  # 小数点7位

def format_price(value):
    """格式化价格 - 去除尾部0"""
    if abs(value) < 1e-7:
        return "0"
    formatted = f"{value:.{PRECISION}f}"
    # 去除尾部0
    formatted = formatted.rstrip('0').rstrip('.')
    return formatted

def v5_logic_predict(window_closes):
    """模拟 V5 模型的预测逻辑"""
    try:
        window = np.array(window_closes, dtype=float)
        current_price = float(window[-1])
        recent_avg = float(np.mean(window[-5:]))
        past_avg = float(np.mean(window[:5]))
        
        direction = 1.0 if recent_avg > past_avg else -1.0
        
        # 计算波动率
        if len(window) >= 11:
            recent_window = window[-10:]
            prev_prices = window[-11:-1]
            returns = (recent_window - prev_prices) / prev_prices
            volatility = float(np.std(returns))
        else:
            volatility = 0.02
        
        volatility = max(volatility, 0.001)  # 防止波动率为 0
        pred_change = direction * volatility * 0.5
        predicted_price = current_price * (1.0 + pred_change)
        
        return float(predicted_price)
    except Exception as e:
        return None

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
        
        # 提取收盘价
        close_prices = df['Close'].values
        timestamps = df.index
        
        # 滚动回测
        for i in range(25, len(close_prices)-1):
            window = close_prices[i-25:i+1]
            
            # V5 预测
            pred_price = v5_logic_predict(window)
            
            if pred_price is None:
                continue
            
            # 实际的明天价格
            actual_next_day = float(close_prices[i+1])
            
            # 计算残差
            residual = actual_next_day - pred_price
            
            residuals.append(residual)
            actuals.append(actual_next_day)
            preds.append(pred_price)
            dates.append(timestamps[i+1])
        
        if len(residuals) == 0:
            print(" [没有有效数据]")
            return None
        
        residuals = np.array(residuals, dtype=float)
        
        # 计算统计指标
        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals))
        min_residual = float(np.min(residuals))
        max_residual = float(np.max(residuals))
        median_residual = float(np.median(residuals))
        
        # 系统偏差不平衡度
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
        print(f" [错误: {str(e)[:40]}]")
        return None

def interpret_residual(mean_residual, std_residual):
    """
    解读残差的含义
    """
    abs_mean = abs(mean_residual)
    
    if abs_mean < 0.1:
        interpretation = "模型不需调整，非常准"
    elif abs_mean < 0.5:
        interpretation = "小幅偏差，低估/高估很齐"
    elif abs_mean < 1.0:
        interpretation = "中等偏差，建议校正"
    else:
        interpretation = "较大残差，需要校正"
    
    bias_quality = "稳定 (容易修复)" if std_residual < abs_mean else "不稳定 (难改善)"
    
    return f"{interpretation} | {bias_quality}"

# === 主程序 ===
print("\n" + "="*160)
print("残差 (Residual) 分析 - 计算每个币种的系统性偏差 【高精度 7位小数】")
print("="*160)
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

print("\n" + "="*160)

# 打印表头
header_symbol = "币种"
header_mean = "平均残差"
header_std = "标准差"
header_median = "中位数"
header_range = "最小~最大"
header_interp = "为什么"

print(f"{header_symbol:<10} | {header_mean:<25} | {header_std:<25} | {header_median:<25} | {header_range:<45} | {header_interp:<45}")
print("="*160)

if results:
    for res in sorted(results, key=lambda x: abs(x['mean_residual'])):
        symbol_name = res['symbol'].replace('-USD', '')
        mean_res = res['mean_residual']
        std_res = res['std_residual']
        median_res = res['median_residual']
        min_res = res['min_residual']
        max_res = res['max_residual']
        interpretation = interpret_residual(mean_res, std_res)
        
        # 格式化数值 - 7位小数
        mean_str = format_price(mean_res)
        std_str = format_price(std_res)
        median_str = format_price(median_res)
        min_str = format_price(min_res)
        max_str = format_price(max_res)
        
        range_str = f"{min_str}~{max_str}"
        print(f"{symbol_name:<10} | {mean_str:>23}$ | {std_str:>23}$ | {median_str:>23}$ | {range_str:<45} | {interpretation:<45}")

print("="*160)

# 统计汇总
print(f"\n统计汇总:")
if results:
    avg_mean = np.mean([r['mean_residual'] for r in results])
    avg_median = np.median([r['median_residual'] for r in results])
    avg_std = np.mean([r['std_residual'] for r in results])
    
    print(f"  总体平均残差: {format_price(avg_mean)}$")
    print(f"  总体中位数: {format_price(avg_median)}$")
    print(f"  总体标准差: {format_price(avg_std)}$")
    
    if avg_mean > 0.0001:
        print(f"  \u63a8估模型: 整体低估 (\u2193 所有币种平均偏低 {format_price(avg_mean)}$)")
    elif avg_mean < -0.0001:
        print(f"  \u63a8估模型: 整体高估 (\u2191 所有币种平均偏高 {format_price(abs(avg_mean))}$)")
    else:
        print(f"  \u63a8估模型: 完全无偏差")
    
    print(f"  成功评估: {len(results)}/{len(SYMBOLS)} 个币种")

if errors:
    print(f"  失败: {len(errors)} 个币种")

print("\n" + "="*160)
print("\n残差解读指南:")
print("""
  残差 = 实际价 - 预测价
  
  残差 > 0 (正值):    模型整体低估了
  残差 < 0 (负值):    模型整体高估了
  残差 ~ 0:           模型无偏差，异常准确
  
  标准差 (残差的波动范围):  
    - 低 (低于平均值):   残差很稳定 -> 容易校正 (正值总是低估，加上正数)
    - 高 (高于平均值):   残差非常不稳定 -> 想要校正效果草
  
  校正策略: 残差趋势图
    - 正值残差 -> 模型低估 -> 正值修正
    - 負值残差 -> 模型高估 -> 負值修正
""".strip())

print("\n" + "="*160 + "\n")
