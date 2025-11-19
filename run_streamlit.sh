#!/bin/bash

# BirdBox Streamlit App Launcher
# This script starts the Streamlit web interface for bird call detection

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════╗"
echo "║         BirdBox - Bird Call Detection App         ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo -e "${YELLOW}Streamlit is not installed.${NC}"
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo -e "${YELLOW}Warning: models directory not found.${NC}"
    echo "The app will offer to download a default model."
fi

# Get number of model files
MODEL_COUNT=$(find models -name "*.pt" -o -name "*.onnx" -o -name "*.engine" 2>/dev/null | wc -l)
echo -e "${GREEN}Found ${MODEL_COUNT} model(s) in models directory${NC}"

echo ""
echo -e "${BLUE}Starting Streamlit app...${NC}"
echo -e "The app will open in your browser at: ${GREEN}http://localhost:8501${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run streamlit_app.py

