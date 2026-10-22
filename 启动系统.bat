@echo off
chcp 65001 >nul
cd /d "%~dp0"
"C:\Users\lenovo\Desktop\VisionAI_Lab\WPy64-312101\python\python.exe" "%~dp0main.py"
echo.
echo 程序已退出。按任意键关闭窗口...
pause >nul