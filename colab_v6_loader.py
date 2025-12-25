#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Remote Loader for V6 Raw Predictions
Downloads and executes colab_v6_1h_raw_predictions.py from GitHub
"""

import urllib.request

print("Downloading V6 1H Raw Predictions Script...\n")

url = "https://raw.githubusercontent.com/caizongxun/cpb-trading-web/main/colab_v6_1h_raw_predictions.py"

try:
    with urllib.request.urlopen(url) as response:
        script = response.read().decode('utf-8')
    
    print("Downloaded successfully! Executing...\n")
    exec(script)
except Exception as e:
    print(f"Error: {e}")
