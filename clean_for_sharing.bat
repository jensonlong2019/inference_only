@echo off
REM 清理脚本：移除不必要的文件，准备分享 (Windows)

echo 正在清理 inference_only 文件夹...

REM 删除 Python 缓存文件
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f"

REM 删除用户生成的文件
if exist "output\batch_result\*.mp3" del /q "output\batch_result\*.mp3"
if exist "output\batch_result\*.wav" del /q "output\batch_result\*.wav"

REM 删除日志文件
for /r . %%f in (*.log) do @if exist "%%f" del /q "%%f"

REM 删除临时文件
for /r . %%f in (*.tmp) do @if exist "%%f" del /q "%%f"
for /r . %%f in (.DS_Store) do @if exist "%%f" del /q "%%f"

echo ✓ 清理完成！
echo.
echo 现在可以压缩 inference_only 文件夹分享了。

pause

