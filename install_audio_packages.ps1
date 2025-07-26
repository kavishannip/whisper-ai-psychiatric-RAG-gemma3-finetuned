# Install audio processing packages for Whisper AI-Psychiatric
Write-Host "Installing audio processing packages for Whisper AI-Psychiatric..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists and activate it
if (Test-Path "rag_env\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "rag_env\Scripts\Activate.ps1"
}

# Install required audio packages
Write-Host "Installing librosa..." -ForegroundColor Blue
pip install librosa>=0.10.0

Write-Host "Installing soundfile..." -ForegroundColor Blue
pip install soundfile>=0.12.0

Write-Host "Installing audio-recorder-streamlit..." -ForegroundColor Blue
pip install audio-recorder-streamlit>=0.0.8

Write-Host "Installing scipy..." -ForegroundColor Blue
pip install scipy>=1.9.0

Write-Host ""
Write-Host "Audio packages installation completed!" -ForegroundColor Green
Write-Host "You can now use speech-to-text and text-to-speech features." -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"
