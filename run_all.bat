@echo off
REM CPB Trading Web - V2 + V5 雙版本並行啟動脚本 (Windows)

echo.
echo ============================================
echo    CPB Trading Web - V2 + V5 並行啟動
echo ============================================
echo.

REM 獲取當前目錄
set SCRIPT_DIR=%CD%
echo ✓ 當前目錄: %SCRIPT_DIR%
echo.

REM 棂測 Python 是否存在
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ✗ 錯誤: 找不到 Python
    echo 請律安裝 Python 3.8+
    pause
    exit /b 1
)

python --version
echo ✓ Python 已找到
echo.

REM 選擇启動方式
echo 選擇启動方法:
echo 1. 三個不同端口並行運行 (推荐測試)
echo 2. 合併為一個 FastAPI 應用程徏
echo 3. 退出
echo.
set /p choice="輸入選擇 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 啟動模式 1: 三個不同端口並行運行
    echo • V2 Server (5000): http://localhost:5000
    echo • V5 API (8001): http://localhost:8001/docs
    echo • V5 Frontend (5001): http://localhost:5001
    echo.
    
    REM 棂測关键 Python 模块
    echo 棂測 Python 模块...
    python -c "import fastapi" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ 錯誤: 找不到需要的 Python 模块
        echo.
        echo 伺與执行以下安裝命令:
        echo pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    
    echo ✓ 所有依賴已找到
    echo.
    
    REM 徟下毋欵
    echo 正在啟動 3 个服務...
    echo.
    
    REM 第一個終端: V2 Server
    start "V2 Server (5000)" cmd /k "cd /d %SCRIPT_DIR% && python -m http.server 5000"
    timeout /t 2 /nobreak
    
    REM 第二个終端: V5 API
    start "V5 API (8001)" cmd /k "cd /d %SCRIPT_DIR% && python app_v5.py"
    timeout /t 2 /nobreak
    
    REM 第三个終端: V5 Frontend
    start "V5 Frontend (5001)" cmd /k "cd /d %SCRIPT_DIR% && python -m http.server 5001"
    timeout /t 2 /nobreak
    
    echo.
    echo ============================================
    echo ✓ 所有服務已啟動！
    echo ============================================
    echo.
    echo • V2 Website: http://localhost:5000
    echo • V5 API Docs: http://localhost:8001/docs
    echo • V5 Website: http://localhost:5001
    echo.
    echo 翻上你的一個檔案但毋器並訪敏！
    echo.
    pause
) else if "%choice%"=="2" (
    echo.
    echo 啟動模式 2: FastAPI 應用程徏 (合併模式)
    echo.
    echo • 正在啟動 FastAPI (港口 8001)...
    echo • 訪啊: http://localhost:8001/docs
    echo.
    
    cd /d %SCRIPT_DIR%
    python app_v5.py
) else if "%choice%"=="3" (
    echo.
    echo 退出
    exit /b 0
) else (
    echo ✗ 無效的選擇
    pause
    exit /b 1
)

REM 結束
echo.
pause
