#!/bin/bash
# run_app.sh

echo "🚀 Starting MedGemma RAG Chatbot Streamlit App..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
fi

# Check if requirements are installed
echo "🔍 Checking requirements..."
pip install -r requirements.txt

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Run the Streamlit app
echo "🌐 Launching Streamlit app at http://localhost:8501"
streamlit run streamlit_app.py