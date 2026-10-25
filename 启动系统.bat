@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 如果不想装全局 Python，可以把便携版解释器路径写进 MT_PYTHON 环境变量再运行
if defined MT_PYTHON (
    "%MT_PYTHON%" main.py
) else (
    python main.py
)

echo.
echo 程序已退出。按任意键关闭窗口...
pause >nul
