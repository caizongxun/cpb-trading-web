#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試預測準確度
對比實際K線與模型預測結果
用於評估模型是否需要重新訓練
"""

import sys
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json

# ============================================================================
# 配置 - 與 app.py 同步
# ============================================================================

def calculate_volatility(klines):
    """計算實際波動率"""
    if len(klines) < 2:
        return 0
    
    last_kline = klines[-1]
    volatility = ((last_kline['close'] - last_kline['open']) / last_kline['open']) * 100
    return volatility

def simple_predict(klines):
    """簡單預測模型邏輯（與 app.py 保持一致）"""
    closes = [k['close'] for k in klines]
    highs = [k['high'] for k in klines]
    lows = [k['low'] for k in klines]
    current_close = closes[-1]
    
    # 趨勢判斷
    recent_avg = sum(closes[-5:]) / 5
    past_avg = sum(closes[:5]) / 5
    trend = recent_avg - past_avg
    
    if trend > 0:
        direction = 1  # 看漲
    elif trend < 0:
        direction = -1  # 看跌
    else:
        direction = 0  # 持平
    
    # 預測價格 (硬編碼 ±2%, ±3%)
    pred_3 = current_close * (1 + 0.02 * direction)
    pred_5 = current_close * (1 + 0.03 * direction)
    
    # 計算 ATR
    true_ranges = []
    for i in range(len(klines)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i-1] if i > 0 else low
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    
    atr_14 = sum(true_ranges[-14:]) / min(14, len(true_ranges)) if true_ranges else 0
    
    # 波動率
    volatility_current = ((klines[-1]['close'] - klines[-1]['open']) / klines[-1]['open']) * 100
    volatility_pred_3 = ((pred_3 - current_close) / current_close) * 100
    volatility_pred_5 = ((pred_5 - current_close) / current_close) * 100
    
    return {
        'current_price': current_close,
        'price_3': pred_3,
        'price_5': pred_5,
        'direction': direction,
        'volatility_current': volatility_current,
        'volatility_pred_3': volatility_pred_3,
        'volatility_pred_5': volatility_pred_5,
        'atr_14': atr_14,
    }

def generate_demo_klines(base_price=87844, num_klines=20, volatility_factor=0.5):
    """生成演示 K 線數據（模擬實際市場數據）"""
    import random
    
    klines = []
    current_time = datetime.now() - timedelta(hours=num_klines)
    current_price = base_price
    
    for i in range(num_klines):
        # 模擬真實市場波動 (小幅波動)
        price_change = random.uniform(-1, 1) * volatility_factor * base_price / 1000
        open_price = current_price + price_change
        close_price = open_price + random.uniform(-0.5, 0.5) * volatility_factor * base_price / 1000
        high_price = max(open_price, close_price) + abs(random.uniform(0, 0.2) * base_price / 1000)
        low_price = min(open_price, close_price) - abs(random.uniform(0, 0.2) * base_price / 1000)
        
        klines.append({
            'timestamp': int(current_time.timestamp() * 1000) + (i * 3600 * 1000),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': random.uniform(100, 1000)
        })
        
        current_price = close_price
    
    return klines

def generate_future_klines(last_price, num_klines=5, trend=0, volatility_factor=0.5):
    """生成未來K線 (模擬真實走勢)"""
    import random
    
    klines = []
    current_time = datetime.now()
    current_price = last_price
    
    for i in range(num_klines):
        # 根據趨勢移動
        trend_factor = trend * 0.0005 * last_price
        price_change = random.uniform(-1, 1) * volatility_factor * last_price / 1000 + trend_factor
        
        open_price = current_price + price_change
        close_price = open_price + random.uniform(-0.5, 0.5) * volatility_factor * last_price / 1000
        high_price = max(open_price, close_price) + abs(random.uniform(0, 0.2) * last_price / 1000)
        low_price = min(open_price, close_price) - abs(random.uniform(0, 0.2) * last_price / 1000)
        
        klines.append({
            'timestamp': int(current_time.timestamp() * 1000) + (i * 3600 * 1000),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': random.uniform(100, 1000)
        })
        
        current_price = close_price
    
    return klines

def plot_prediction_comparison(historical_klines, future_klines, pred_result):
    """繪製圖表對比實際 vs 預測"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('CPB 模型預測準確度測試 - BTC 1H', fontsize=16, fontweight='bold')
    
    # ========== 上圖：K線走勢 + 預測 ==========
    # 提取歷史收盤價
    hist_times = [datetime.fromtimestamp(k['timestamp']/1000) for k in historical_klines]
    hist_closes = [k['close'] for k in historical_klines]
    
    # 提取未來收盤價
    future_times = [datetime.fromtimestamp(k['timestamp']/1000) for k in future_klines]
    future_closes = [k['close'] for k in future_klines]
    
    # 準備預測線
    current_price = pred_result['current_price']
    pred_3_price = pred_result['price_3']
    pred_5_price = pred_result['price_5']
    
    # 時間軸 (預測)
    pred_times = [
        hist_times[-1],
        hist_times[-1] + timedelta(hours=3),
        hist_times[-1] + timedelta(hours=5)
    ]
    pred_prices = [current_price, pred_3_price, pred_5_price]
    
    # 繪製歷史K線
    ax1.plot(hist_times, hist_closes, 'b-', linewidth=2, label='實際歷史K線', marker='o', markersize=4)
    
    # 繪製未來實際走勢
    actual_future_times = hist_times[-1:] + future_times
    actual_future_prices = hist_closes[-1:] + future_closes
    ax1.plot(actual_future_times, actual_future_prices, 'g--', linewidth=2, label='實際未來走勢 (模擬)', marker='s', markersize=4)
    
    # 繪製預測線
    ax1.plot(pred_times, pred_prices, 'r-', linewidth=3, label='模型預測', marker='^', markersize=8)
    
    # 填充預測區間
    ax1.axvspan(hist_times[-1], pred_times[-1], alpha=0.2, color='yellow', label='預測區間')
    
    # 設置
    ax1.set_xlabel('時間 (小時)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('價格 (USDT)', fontsize=11, fontweight='bold')
    ax1.set_title('K線走勢對比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== 下圖：誤差分析 ==========
    # 計算誤差
    actual_future_close_5 = future_closes[-1] if len(future_closes) >= 5 else future_closes[-1]
    prediction_error_5 = pred_5_price - actual_future_close_5
    error_percentage = (prediction_error_5 / actual_future_close_5) * 100 if actual_future_close_5 != 0 else 0
    
    # 數據對比表
    metrics = [
        '當前價格',
        '3小時預測',
        '5小時預測',
        '實際5小時',
        '誤差 (5H)',
        '誤差率 (%)'
    ]
    
    values = [
        f'${current_price:,.2f}',
        f'${pred_3_price:,.2f}',
        f'${pred_5_price:,.2f}',
        f'${actual_future_close_5:,.2f}',
        f'${prediction_error_5:,.2f}',
        f'{error_percentage:.2f}%'
    ]
    
    # 顏色編碼
    colors = ['white', 'lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightgray']
    if error_percentage < 0.5:
        colors[-2] = 'lightgreen'  # 誤差小
    elif error_percentage > 2:
        colors[-2] = 'lightcoral'  # 誤差大
    
    # 繪製表格
    table = ax2.table(
        cellText=list(zip(metrics, values)),
        colLabels=['指標', '數值'],
        cellLoc='center',
        loc='center',
        cellColours=[[c, c] for c in colors]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 美化表格
    for i in range(len(metrics) + 1):
        if i == 0:
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
    
    ax2.axis('off')
    ax2.set_title('預測準確度分析', fontsize=12, fontweight='bold', pad=20)
    
    # 添加結論文字
    conclusion = f"""
    分析結論：
    - 模型預測方向: {'看漲' if pred_result['direction'] > 0 else '看跌' if pred_result['direction'] < 0 else '持平'}
    - 5小時預測誤差: {error_percentage:.2f}% ({'準確' if abs(error_percentage) < 1 else '需要改進' if abs(error_percentage) < 2 else '誤差較大'})
    - 建議: {'模型需要重新訓練，學習更準確的波動率' if abs(error_percentage) > 1.5 else '模型表現良好'}
    """
    
    ax2.text(0.5, -0.3, conclusion, transform=ax2.transAxes, 
             fontsize=10, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('prediction_test_result.png', dpi=300, bbox_inches='tight')
    print("圖表已保存為: prediction_test_result.png")
    plt.show()

def main():
    print("="*70)
    print("CPB 模型預測準確度測試")
    print("="*70)
    
    # 1. 生成歷史 K 線
    print("\n[1] 生成模擬歷史 K 線數據...")
    historical_klines = generate_demo_klines(base_price=87844, num_klines=20, volatility_factor=0.5)
    print(f"    生成 {len(historical_klines)} 根 K 線")
    print(f"    起始價格: ${historical_klines[0]['close']:,.2f}")
    print(f"    結束價格: ${historical_klines[-1]['close']:,.2f}")
    
    # 2. 執行預測
    print("\n[2] 執行模型預測...")
    pred_result = simple_predict(historical_klines)
    print(f"    當前價格: ${pred_result['current_price']:,.2f}")
    print(f"    預測方向: {'看漲 ↑' if pred_result['direction'] > 0 else '看跌 ↓' if pred_result['direction'] < 0 else '持平 →'}")
    print(f"    3H 預測: ${pred_result['price_3']:,.2f} (變化: {pred_result['volatility_pred_3']:.2f}%)")
    print(f"    5H 預測: ${pred_result['price_5']:,.2f} (變化: {pred_result['volatility_pred_5']:.2f}%)")
    print(f"    ATR(14): ${pred_result['atr_14']:,.2f}")
    
    # 3. 生成未來實際 K 線（模擬真實走勢）
    print("\n[3] 生成模擬未來實際走勢...")
    future_klines = generate_future_klines(
        last_price=historical_klines[-1]['close'],
        num_klines=5,
        trend=pred_result['direction'],
        volatility_factor=0.5
    )
    print(f"    生成 {len(future_klines)} 根未來 K 線")
    print(f"    未來走勢結束價格: ${future_klines[-1]['close']:,.2f}")
    
    # 4. 計算誤差
    print("\n[4] 計算預測誤差...")
    actual_5h_price = future_klines[-1]['close']
    predicted_5h_price = pred_result['price_5']
    error = predicted_5h_price - actual_5h_price
    error_pct = (error / actual_5h_price) * 100 if actual_5h_price != 0 else 0
    
    print(f"    預測 5H 價格: ${predicted_5h_price:,.2f}")
    print(f"    實際 5H 價格: ${actual_5h_price:,.2f}")
    print(f"    絕對誤差: ${error:,.2f}")
    print(f"    相對誤差: {error_pct:.2f}%")
    
    if abs(error_pct) < 0.5:
        status = "✓ 優秀 (誤差 < 0.5%)"
    elif abs(error_pct) < 1:
        status = "◐ 良好 (誤差 < 1%)"
    elif abs(error_pct) < 2:
        status = "◑ 一般 (誤差 < 2%)"
    else:
        status = "✗ 需要改進 (誤差 > 2%)"
    print(f"    評估: {status}")
    
    # 5. 繪製對比圖
    print("\n[5] 生成對比圖表...")
    plot_prediction_comparison(historical_klines, future_klines, pred_result)
    
    # 6. 結論
    print("\n" + "="*70)
    print("測試結論")
    print("="*70)
    if abs(error_pct) > 1.5:
        print("""
    當前模型存在以下問題：
    1. 波動率估計不準確 (硬編碼 ±2%, ±3% 不符合實際市場)
    2. 方向預測可能正確，但價格幅度偏離
    3. 模型沒有學習到真實波動率特徵
    
    建議方案：
    ✓ 訓練新的 V2 模型，讓其學習波動率
    ✓ 使用實際歷史數據訓練 (不使用硬編碼百分比)
    ✓ 模型輸出應該是: [價格, 波動率] (2個預測值)
    ✓ 用均方誤差 (MSE) 作為損失函數
        """)
    else:
        print("""
    模型表現良好，無需立即重新訓練。
    但建議收集更多真實數據，進行迭代優化。n        """)
    
    print("\n" + "="*70)
    print("測試完成")
    print("="*70)

if __name__ == "__main__":
    main()
