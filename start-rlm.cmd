@echo off
cd /d "%~dp0"
py -3 -c "import launcher,sys;sys.exit(2 if launcher.health(8080) else 0)"
if errorlevel 3 goto failed
if errorlevel 2 (
  echo A model server is already running. Close it before starting the WSL server.
  pause
  exit /b 2
)
if errorlevel 1 goto failed
wsl.exe -d LocalCodingAssistant -u coder --cd /home/coder/local-coding-assistant -- .cache/rlm-env/bin/python wsl_server.py --model "%~dp0.cache\models\Qwen3.8-27B-UD-Q4_K_M.gguf" %*
if errorlevel 1 goto failed
exit /b 0
:failed
pause
exit /b 1
