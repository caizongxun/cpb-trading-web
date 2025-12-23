@echo off
REM CPB Trading Web - V2 Model
REM 快速啟動脚本

echo ================================================================================
echo            CPB Trading Web - V2 Model
echo ================================================================================
echo.
echo 正在啟動 FastAPI Server...
echo.
echo API 地址: http://localhost:8000
echo 前端地址: http://localhost:5000
echo API 文件: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止 Server
echo ================================================================================
echo.

python app.py

pause
