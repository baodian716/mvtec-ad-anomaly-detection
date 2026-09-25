@echo off
rem 雙擊啟動 MVTec AD demo：後端 FastAPI + ONNX Runtime，前端為 demo/frontend/dist
rem 啟動後會自動開啟瀏覽器（http://localhost:7860）；關閉這個視窗即停止 demo
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" demo\backend\server.py
pause
