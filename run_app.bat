@echo off
REM run_app.bat

echo 🚀 Starting MedGemma RAG Chatbot Streamlit App...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if requirements are installed
echo 🔍 Installing/checking requirements...
pip install -r requirements.txt

REM Create .streamlit directory if it doesn't exist
if not exist ".streamlit" mkdir .streamlit

REM Run the Streamlit app
echo 🌐 Launching Streamlit app at http://localhost:8501
streamlit run streamlit_app.py

pause