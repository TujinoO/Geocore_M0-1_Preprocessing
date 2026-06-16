@echo off
setlocal
cd /d "%~dp0"
echo Starting Geo-Core AI foreground mask API...
echo OpenAPI docs: http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
endlocal
