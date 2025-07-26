@echo off
echo Installing audio processing packages for Whisper AI-Psychiatric...
echo.

:: Activate virtual environment if it exists
if exist "rag_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call rag_env\Scripts\activate.bat
)

:: Install required audio packages
echo Installing librosa...
pip install librosa>=0.10.0

echo Installing soundfile...
pip install soundfile>=0.12.0

echo Installing audio-recorder-streamlit...
pip install audio-recorder-streamlit>=0.0.8

echo Installing scipy...
pip install scipy>=1.9.0

echo.
echo Audio packages installation completed!
echo You can now use speech-to-text and text-to-speech features.
echo.
pause
