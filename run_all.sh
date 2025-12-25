#!/bin/bash

# CPB Trading Web - V2 + V5 雙版本並行啟動脚本

echo ""
echo "=============================================="
echo "   CPB Trading Web - V2 + V5 並行啟動"
echo "=============================================="
echo ""

# 獲取上一个呪目錄
 SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "✓ 當前目錄: $SCRIPT_DIR"
echo ""

# 棂測依賴
echo "棂測依賴..."

if ! command -v python3 &> /dev/null; then
    echo "✗ 錯誤: 找不到 python3"
    exit 1
fi

echo "✓ Python 已找到: $(python3 --version)"
echo ""

# 選擇启動方式
echo "選擇启動方法:"
echo "1. 三個不同端口並行運行 (推荐測試)"
echo "2. 合併為一個 FastAPI 應用程徏"
read -p "輸入選擇 (1 或 2): " choice

case $choice in
    1)
        echo ""
        echo "啟動模式 1: 三個不同端口並行運行"
        echo "• V2 Server (5000): http://localhost:5000"
        echo "• V5 API (8001): http://localhost:8001/docs"
        echo "• V5 Frontend (5001): http://localhost:5001"
        echo ""
        
        # 棂測 Python 包
        echo "棂測 Python 包..."
        python3 -c "import fastapi" 2>/dev/null || {
            echo "✗ 錯誤: 找不到 fastapi"
            echo "伺與安裝: pip install fastapi uvicorn huggingface-hub tensorflow yfinance pandas numpy"
            exit 1
        }
        
        echo "✓ 所有依賴已找到"
        echo ""
        
        # 開訿終端
if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            echo "在 macOS 上啟動..."
            
            osascript <<EOF
tell app "Terminal"
    create window with default profile
    tell application "System Events" to keystroke "cd '$SCRIPT_DIR' && python3 -m http.server 5000" & key code 36
    tell application "Terminal" to activate
end tell

tell app "Terminal"
    create window with default profile
    tell application "System Events" to keystroke "cd '$SCRIPT_DIR' && python3 app_v5.py" & key code 36
    tell application "Terminal" to activate
end tell

tell app "Terminal"
    create window with default profile
    tell application "System Events" to keystroke "cd '$SCRIPT_DIR' && python3 -m http.server 5001" & key code 36
    tell application "Terminal" to activate
end tell
EOF
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            echo "在 Linux 上啟動..."
            
            # 使用 gnome-terminal 或 xterm
            if command -v gnome-terminal &> /dev/null; then
                gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && python3 -m http.server 5000; exec bash" &
                sleep 2
                gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && python3 app_v5.py; exec bash" &
                sleep 2
                gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && python3 -m http.server 5001; exec bash" &
            elif command -v xterm &> /dev/null; then
                xterm -e "cd '$SCRIPT_DIR' && python3 -m http.server 5000" &
                sleep 2
                xterm -e "cd '$SCRIPT_DIR' && python3 app_v5.py" &
                sleep 2
                xterm -e "cd '$SCRIPT_DIR' && python3 -m http.server 5001" &
            else
                echo "✗ 找不到支持的終端程式"
                exit 1
            fi
        fi
        
        sleep 3
        echo ""
        echo "=============================================="
        echo "✓ 所有服務已啟動！"
        echo "=============================================="
        echo ""
        echo "• V2 Website: http://localhost:5000"
        echo "• V5 API Docs: http://localhost:8001/docs"
        echo "• V5 Website: http://localhost:5001"
        echo ""
        echo "翻上你的但毋器並訪啊！"
        echo ""
        ;;
    
    2)
        echo ""
        echo "啟動模式 2: FastAPI 應用程徏 (合併模式)"
        echo "• 需要修改 app_v5.py"
        echo "• 估計 30 秒後啟動"
        echo ""
        
        # 棂測是否已整合
if grep -q "StaticFiles" "app_v5.py" 2>/dev/null; then
            echo "✓ app_v5.py 已整合静态檔案支援"
            echo "啟動 FastAPI..."
            python3 app_v5.py
        else
            echo "✗ app_v5.py 還未整合静态檔案支援"
            echo "請先修改 app_v5.py ："
            echo ""
            echo "from fastapi.staticfiles import StaticFiles"
            echo "app.mount('/', StaticFiles(directory='.', html=True), name='static')"
            echo ""
            exit 1
        fi
        ;;
    
    *)
        echo "✗ 無效的選擇"
        exit 1
        ;;
esac
