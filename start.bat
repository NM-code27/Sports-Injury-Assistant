@echo off
REM start.bat — Start PhysioAI on Windows

echo.
echo ╔══════════════════════════════════════╗
echo ║         PhysioAI — Starting          ║
echo ╚══════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

REM Check Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama not found. Install from https://ollama.com
    pause
    exit /b 1
)

REM Create venv if needed
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

REM Data and index folders
if not exist "data" mkdir data
if not exist "chroma_db" mkdir chroma_db

set EMBED_MODEL=%OLLAMA_EMBED_MODEL%
if "%EMBED_MODEL%"=="" set EMBED_MODEL=nomic-embed-text
set EMBED_BACKEND=%PHYSIOAI_EMBEDDING_BACKEND%
if "%EMBED_BACKEND%"=="" set EMBED_BACKEND=auto

set PDF_COUNT=0
for /r data %%f in (*.pdf) do set /a PDF_COUNT+=1

echo PDF folder: %CD%\data
if %PDF_COUNT%==0 (
    echo No PDFs found yet. Drop files into data\ and restart.
) else (
    echo Found %PDF_COUNT% PDF file^(s^) in data\
)
echo Embeddings backend: %EMBED_BACKEND% ^(model: %EMBED_MODEL%^)

ollama list | findstr /R /B /C:"%EMBED_MODEL%" >nul
if errorlevel 1 (
    echo Embedding model "%EMBED_MODEL%" is not pulled.
    echo Run: ollama pull %EMBED_MODEL%
)

echo.
echo Starting PhysioAI backend on http://localhost:8000
echo Open your browser at: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

cd backend
uvicorn main:app --reload --port 8000
pause
